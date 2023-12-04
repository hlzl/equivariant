<h1 align='center'>equivariant</h1>

This repository contains the implementation for the ACM AISec '23 paper [Equivariant Differentially Private Deep Learning: Why DP-SGD Needs Sparser Models](https://doi.org/10.1145/3605764.3623902).

## Installation
We recommend [mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) to manage packages. You can create an environment and install all necessary packages with:

```bash
mamba env create -n <env-name> -f requirements.yml
```

Requires Python 3.8+ and CUDA 11.7 (or another compatible version).

## Quick reproduction

There are `config` files available to reproduce the results from the paper. E.g. the Equivariant ResNet-9 on CIFAR100 with a $(\varepsilon = 2, \delta = 10^{-5})$-DP can be run with:

```bash
python train.py setup=eqresnet9-cifar100 setup.dp.target_epsilon=2 setup.dp.target_delta=1e-5
```

## Citation

If you found this library or the paper to be useful, then please cite:

```bibtex
@inproceedings{hoelzl2023equivariant,
author = {H\"{o}lzl, Florian A. and Rueckert, Daniel and Kaissis, Georgios},
title={{E}quivariant {D}ifferentially {P}rivate {D}eep {L}earning: {W}hy {DP-SGD} {N}eeds {S}parser {M}odels},
year = {2023},
isbn = {9798400702600},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3605764.3623902},
doi = {10.1145/3605764.3623902},
booktitle = {Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security},
pages = {11–22},
numpages = {12},
location = {<conf-loc>, <city>Copenhagen</city>, <country>Denmark</country>, </conf-loc>},
series = {AISec '23}
}
```

(Also consider starring the project on GitHub. An improved version might follow soon ;) )
