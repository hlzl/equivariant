import os
import random
import warnings
from collections import defaultdict
from datetime import datetime

warnings.filterwarnings("ignore")
from statistics import mean
import math
import hydra

import numpy as np
import torch
from functorch import grad, make_functional, vmap
from opacus.accountants.utils import get_noise_multiplier
from opacus.optimizers import DPOptimizer
from opacus.utils.batch_memory_manager import wrap_data_loader
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

import networks  # NOTE: this is required for hydra even though not directly used here
from data import CustomDataloader
from utils import PrintAndSaveFile


def accuracy(preds, targets):
    """For one hot encoded labels."""
    return torch.sum(preds.argmax(dim=-1) == targets) / len(targets)


def compute_loss(model, params, sample, target, loss_fn):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)
    predictions = model(params, batch)
    loss = loss_fn(predictions, targets)
    return loss


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    # Log setup
    print(
        f"SETUP:\nNetwork: "
        f"{cfg.setup.network._target_.split('.')[-1]}\tDataset: {cfg.setup.dataset.name}"
        f"\tDP: ({cfg.setup.dp.target_epsilon}, {cfg.setup.dp.target_delta})\n"
    )

    compute_per_sample_grads = vmap(
        vmap(grad(compute_loss, argnums=1), in_dims=(None, None, 0, 0, None)),
        in_dims=(None, None, 0, 0, None),
    )  # nested vmap for data augmentation

    # Setting up training
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if cfg.hardware.deterministic:
        print("Determinism makes things slower.")
        SEED = 9504
        torch.backends.cudnn.deterministic = True
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        cpu_generator = torch.Generator(device="cpu").manual_seed(SEED)
        gpu_generator = torch.Generator(device=device).manual_seed(SEED)
    else:
        SEED = None
        cpu_generator = torch.Generator(device="cpu")
        gpu_generator = torch.Generator(device=device)

    # Create dataset
    dataset = CustomDataloader(cfg)
    bs = cfg.setup.params.bs
    epochs = int(cfg.setup.params.update_steps / (len(dataset.train_ds) / bs))

    # Caculate required noise
    clip_norm = cfg.setup.dp.clip_norm
    noise = get_noise_multiplier(
        target_epsilon=cfg.setup.dp.target_epsilon,
        target_delta=cfg.setup.dp.target_delta,
        sample_rate=bs / len(dataset.train_ds),
        epochs=epochs,
        accountant="rdp",
    )

    # Adapt BS to fit on GPU with augmentation multiplicities and create dataloader
    if bs * dataset.aug_mult > cfg.hardware.bs_physical:
        bs_physical = cfg.hardware.bs_physical // dataset.aug_mult
    else:
        bs_physical = bs

    train_loader, val_loader, test_loader = dataset.get_dataloader(
        bs, bs_physical, cpu_generator
    )

    # Create model and make it functional
    try:
        model = hydra.utils.instantiate(
            cfg.setup.network,
            input_channels=dataset.n_channels,
            num_classes=dataset.n_classes,
        ).to(device)
    except:
        raise ValueError(f"Model {cfg.setup.network} is not known.")
    fmodel, params = make_functional(model)
    forward = lambda x: fmodel(params, x)

    # Optimization
    ema = ExponentialMovingAverage(params, decay=cfg.setup.params.decay)
    optim = DPOptimizer(
        hydra.utils.instantiate(
            cfg.setup.optim.optimizer, params, lr=cfg.setup.optim.optimizer.lr
        ),
        noise_multiplier=noise,
        max_grad_norm=clip_norm,
        expected_batch_size=bs,
        loss_reduction=cfg.setup.optim.reduction,
        generator=gpu_generator,
    )

    criterion = torch.nn.CrossEntropyLoss(reduction=cfg.setup.optim.reduction)
    if cfg.setup.optim.lr_reduceonplateau:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            mode="min",
            factor=0.5,
            verbose=True,
            patience=cfg.setup.optim.lr_patience,
        )

    numel = sum(p.numel() for p in params)
    print(f"Number of parameters: {numel}\t Noise multiplier: {noise:.2f}")

    # Wrap dataloader for accumulating gradients and adapt number of steps
    if bs_physical < bs:
        train_loader = wrap_data_loader(
            data_loader=train_loader,
            max_batch_size=bs_physical,
            optimizer=optim,
        )
        total_steps = math.ceil(len(dataset.train_ds) / bs_physical)
    else:
        total_steps = len(train_loader)

    # It's finally time to start the training loooop
    torch.set_grad_enabled(False)
    for epoch in range(epochs):
        model.train()
        train_metrics = defaultdict(list)
        for batch, target in tqdm(
            train_loader, leave=False, desc=f"Epoch {epoch+1}", total=total_steps
        ):
            optim.zero_grad(True)
            # Batch is augs, N, C, H, W - Labels are now augs, N
            # NOTE: Without augs, this dimension is still unsqueezed in the dataloader
            batch = batch.permute(1, 0, 2, 3, 4).to(device)
            target = (
                torch.tile(target, (dataset.aug_mult,))
                .view(dataset.aug_mult, len(target))
                .to(device)
            )

            per_sample_grads = compute_per_sample_grads(
                fmodel, params, batch, target, criterion
            )

            # Update grads of params with per sample grads - average if augs were used
            for param, grad_sample in zip(params, per_sample_grads):
                grad_sample = grad_sample.sum(0)  # augmentation multiplicity
                param.grad_sample = grad_sample.contiguous()

            # Optimize model and update moving average with new parameters
            optim.step()
            ema.update()

            # Evaluate train performance
            y_pred = forward(batch[0, ...])
            train_metrics["loss"].append(criterion(y_pred, target[0]).item())
            train_metrics["acc"].append(accuracy(y_pred, target[0]).item())

        # Validation set
        model.eval()
        if val_loader is not None:
            val_metrics = defaultdict(list)
            with ema.average_parameters():
                for batch, target in tqdm(val_loader, leave=False):
                    y_pred = forward(batch.to(device)).detach()
                    val_metrics["loss"].append(
                        criterion(y_pred, target.to(device)).item()
                    )
                    val_metrics["acc"].append(
                        accuracy(y_pred, target.to(device)).item()
                    )

            print(f"Epoch {epoch+1}")
            print(
                f"\tTrain Loss: {mean(train_metrics['loss']):.4f}"
                f"\tTrain Accuracy: {mean(train_metrics['acc']):.4f}"
            )
            print(
                f"\tValid. Loss: {mean(val_metrics['loss']):.4f}"
                f"\tValid. Accuracy: {mean(val_metrics['acc']):.4f}"
            )

        if cfg.setup.optim.lr_reduceonplateau:
            scheduler.step(mean(val_metrics["loss"]))

    # Test set
    if test_loader is not None:
        test_metrics = defaultdict(list)
        with ema.average_parameters():
            for batch, target in tqdm(test_loader, leave=False):
                y_pred = forward(batch.to(device)).detach()
                test_metrics["loss"].append(criterion(y_pred, target.to(device)).item())
                test_metrics["acc"].append(accuracy(y_pred, target.to(device)).item())

        print(
            f"\tTest Loss: {mean(test_metrics['loss']):.4f}"
            f"\tTest Accuracy: {mean(test_metrics['acc']):.4f}"
        )


if __name__ == "__main__":
    if not os.path.exists("logs"):
        os.makedirs("logs")
    with PrintAndSaveFile(
        f"logs/{datetime.now().strftime('%Y%m%d-%H%M')}-out.txt"
    ) as f:
        main()
