import numpy as np
import torch

from typing import List, Union, Tuple, Callable, Dict, Iterable

from .basis import KernelBasis
from .steerable_filters_basis import SteerableFiltersBasis
from equivariant.group_theory import *
from equivariant.group_theory.representations import change_basis


class GaussianRadialProfile(KernelBasis):
    def __init__(self, radii: List[float], sigma: Union[List[float], float]):
        r"""
        Basis for kernels defined over a radius in :math:`\R^+_0`.
        Each basis element is defined as a Gaussian function.
        Different basis elements are centered at different radii (``rings``) and can possibly be associated with
        different widths (``sigma``).
        More precisely, the following basis is implemented:
        .. math::
            \mathcal{B} = \left\{ b_i (r) :=  \exp \left( \frac{ \left( r - r_i \right)^2}{2 \sigma_i^2} \right) \right\}_i
        In order to build a complete basis of kernels, you should combine this basis with a basis which defines the
        angular profile, see for example :class:`~SphericalShellsBasis` or
        :class:`~CircularShellsBasis`.
        Args:
            radii (list): centers of each basis element. They should be different and spread to cover all
                domain of interest
            sigma (list or float): widths of each element. Can potentially be different.
        """

        if isinstance(sigma, float):
            sigma = [sigma] * len(radii)

        assert len(radii) == len(sigma)
        assert isinstance(radii, list)

        for r in radii:
            assert r >= 0.0

        for s in sigma:
            assert s > 0.0

        super(GaussianRadialProfile, self).__init__(len(radii), (1, 1))

        self.register_buffer(
            "radii", torch.tensor(radii, dtype=torch.float32).reshape(1, -1, 1, 1)
        )
        self.register_buffer(
            "sigma", torch.tensor(sigma, dtype=torch.float32).reshape(1, -1, 1, 1)
        )

    def sample(self, radii: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        r"""
        Sample the continuous basis elements on the discrete set of radii in ``radii``.
        Optionally, store the resulting multidimentional array in ``out``.
        ``radii`` must be an array of shape `(N, 1)`, where `N` is the number of points.
        Args:
            radii (~torch.Tensor): radii where to evaluate the basis elements
            out (~torch.Tensor, optional): pre-existing array to use to store the output
        Returns:
            the sampled basis
        """
        assert len(radii.shape) == 2
        assert radii.shape[1] == 1
        S = radii.shape[0]

        if out is None:
            out = torch.empty(
                (S, self.dim, self.shape[0], self.shape[1]),
                device=radii.device,
                dtype=radii.dtype,
            )

        assert out.shape == (S, self.dim, self.shape[0], self.shape[1])

        radii = radii.reshape(-1, 1, 1, 1)

        assert not torch.isnan(radii).any()

        d = (self.radii - radii) ** 2

        if radii.requires_grad:
            out[:] = torch.exp(-0.5 * d / self.sigma**2)
        else:
            out = torch.exp(-0.5 * d / self.sigma**2, out=out)

        return out

    def __getitem__(self, r):
        assert r < self.dim
        return {
            "radius": self.radii[0, r, 0, 0].item(),
            "sigma": self.sigma[0, r, 0, 0].item(),
            "idx": r,
        }

    def __eq__(self, other):
        if isinstance(other, GaussianRadialProfile):
            return torch.allclose(
                self.radii, other.radii.to(self.radii.device)
            ) and torch.allclose(self.sigma, other.sigma.to(self.sigma.device))
        else:
            return False

    def __hash__(self):
        return hash(self.radii.cpu().numpy().tobytes()) + hash(
            self.sigma.cpu().numpy().tobytes()
        )


class CircularShellsBasis(SteerableFiltersBasis):
    def __init__(
        self,
        L: int,
        radial: GaussianRadialProfile,
        filter: Callable[[Dict], bool] = None,
        axis: float = np.pi / 2,
    ):
        r"""
        Build the tensor product basis of a radial profile basis and a circular harmonics basis for kernels over the
        Euclidean space :math:`\R^2`.
        The kernel space is spanned by an independent basis for each shell.
        The kernel space each shell is spanned by the circular harmonics of frequency up to `L`
        (an independent copy of each for each cell).
        Given the bases :math:`A = \{a_j\}_j` for the circular shells and
        :math:`D = \{d_r\}_r` for the radial component (indexed by :math:`r \geq 0`, the radius different rings),
        this basis is defined as
        .. math::
            C = \left\{c_{i,j}(\bold{p}) := d_r(||\bold{p}||) a_j(\hat{\bold{p}}) \right\}_{r, j}
        where :math:`(||\bold{p}||, \hat{\bold{p}})` are the polar coordinates of the point
        :math:`\bold{p} \in \R^n`.
        The radial component is parametrized using :class:`~GaussianRadialProfile`.
        Args:
            L (int): the maximum circular frequency
            radial (GaussianRadialProfile): the basis for the radial profile
            filter (callable, optional): function used to filter out some basis elements. It takes as input a dict
                describing a basis element and should return a boolean value indicating whether to keep (`True`) or
                discard (`False`) the element. By default (`None`), all basis elements are kept.
        Attributes:
            ~.radial (GaussianRadialProfile): the radial basis
            ~.L (int): the maximum circular frequency
        """

        self.L: int = L

        assert isinstance(radial, GaussianRadialProfile)

        self._angular_dim = 2 * L + 1

        # number of invariant subspaces
        self._num_inv_spaces = 0

        G = O2(L)

        if filter is not None:
            _filter = torch.zeros(self._angular_dim * len(radial), dtype=torch.bool)

            js = []
            _idx_map = []
            _steerable_idx_map = []
            i = 0
            steerable_i = 0
            for j in range(self.L + 1):
                attr2 = {
                    "irrep:" + k: v
                    for k, v in G.irrep(int(j > 0), j).attributes.items()
                }
                attr2["j"] = (int(j > 0), j)  # the id of the O(2) irrep

                dim = 2 if j > 0 else 1

                multiplicity = 0

                for attr1 in radial:
                    attr = dict()
                    attr.update(attr1)
                    attr.update(attr2)

                    if filter(attr):
                        multiplicity += 1
                        _filter[i : i + dim] = 1
                        _idx_map += list(range(i, i + dim))
                        _steerable_idx_map.append(steerable_i)

                    i += dim
                    steerable_i += 1

                js.append(((int(j > 0), j), multiplicity))  # the O(2) irrep ID
                self._num_inv_spaces += multiplicity

            self._idx_map = np.array(_idx_map)
            self._steerable_idx_map = np.array(_steerable_idx_map)
        else:
            _filter = None
            self._idx_map = None
            self._steerable_idx_map = None
            js = [
                ((int(j > 0), j), len(radial))  # the O(2) irrep ID
                for j in range(L + 1)
            ]

        self.axis = axis

        action = G.standard_representation()
        action = change_basis(
            action,
            action(G.element((0, axis))),
            name=f"StandardAction|axis=[{axis}]",
        )
        super(CircularShellsBasis, self).__init__(G, action, js)

        self.radial = radial

        if _filter is None:
            self._filter = None
        else:
            self.register_buffer("_filter", _filter)

    def sample(self, points: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        r"""
        Sample the continuous basis elements on a discrete set of ``points`` in the space :math:`\R^n`.
        Optionally, store the resulting multidimensional array in ``out``.
        ``points`` must be an array of shape `(N, 2)` containing `N` points in the space.
        Note that the points are specified in cartesian coordinates :math:`(x, y, z, ...)`.
        Args:
            points (~torch.Tensor): points in the n-dimensional Euclidean space where to evaluate the basis elements
            out (~torch.Tensor, optional): pre-existing array to use to store the output
        Returns:
            the sampled basis
        """
        assert len(points.shape) == 2
        assert points.shape[1] == self.dimensionality, (
            points.shape,
            self.dimensionality,
        )

        S = points.shape[0]

        radii = torch.norm(points, dim=1, keepdim=True)

        non_origin_mask = (radii > 1e-9).reshape(-1)
        sphere = points[non_origin_mask, :] / radii[non_origin_mask, :]

        if out is None:
            out = torch.empty(
                S, self.dim, 1, 1, device=points.device, dtype=points.dtype
            )

        assert out.shape == (S, self.dim, 1, 1)

        # sample the radial basis
        radial = self.radial.sample(radii)
        assert radial.shape[-2:] == (1, 1)
        radial = radial[..., 0, 0]

        assert not torch.isnan(radial).any()

        # sample the angular basis
        circular = torch.empty(
            S, self._angular_dim, device=points.device, dtype=points.dtype
        )

        # where r>0, we sample all frequencies
        circular[non_origin_mask, :] = self.circular_harmonics(
            sphere, self.L, phase=self.axis
        )

        # only frequency 0 is sampled at the origin. Other frequencies are set to 0
        circular[~non_origin_mask, :1] = 1.0

        # This trick allows us to compute meaningful gradients at the origin
        circular[~non_origin_mask, 1:] = 0.0

        tensor_product = torch.einsum("pa,pb->pab", radial, circular)

        n_radii = len(self.radial)

        if self._filter is None:
            tmp_out = out
        else:
            tmp_out = torch.empty(
                S,
                self._angular_dim * n_radii,
                1,
                1,
                device=points.device,
                dtype=points.dtype,
            )

        for j in range(self.L + 1):
            dim = 2 if j > 0 else 1
            last = 2 * j + 1
            first = last - dim
            tmp_out[:, first * n_radii : last * n_radii, 0, 0].view(S, n_radii, dim)[
                :
            ] = tensor_product[:, :, first:last]

        if self._filter is not None:
            out[:] = tmp_out[:, self._filter, ...]

        return out

    def circular_harmonics(self, points: torch.Tensor, L: int, phase: float = 0.0):
        r"""
        Compute the circular harmonics up to frequency ``L``.
        """
        assert len(points.shape) == 2
        assert points.shape[1] == 2

        device = points.device
        dtype = points.dtype

        S = points.shape[0]

        x, y = points.T

        angles = torch.atan2(y, x).view(S, 1) - phase

        freqs = torch.arange(1, L + 1, device=device, dtype=dtype).view(1, L)

        freqs_times_angles = freqs * angles

        del freqs, angles

        Y = torch.empty((S, 2 * L + 1), dtype=dtype, device=device)

        Y[:, 0] = 1.0
        Y[:, 1::2] = torch.cos(freqs_times_angles)
        Y[:, 2::2] = torch.sin(freqs_times_angles)

        return Y

    def steerable_attrs_j_iter(self, j: Tuple) -> Iterable:
        # The attributes don't describe a single basis element but a group of basis elements which span
        # an invariant subspace. This is needed to generate the attributes of the SteerableKernelBasis.
        j_id = j
        f, j = j

        if f != int(j > 0):
            return

        idx = sum(self.multiplicity((int(_j > 0), _j)) for _j in range(j))
        dim = 2 if j > 0 else 1
        i = 0

        attr1 = {"irrep:" + k: v for k, v in self.group.irrep(*j_id).attributes.items()}
        # since this methods return iterables of attributes built on the fly, load all attributes first
        # and then iterate on these lists
        radial_attrs = list(self.radial)

        for radial_idx, attr2 in enumerate(radial_attrs):
            if self._filter is None or (self._filter[i : i + dim] == 1).all():
                assert attr2["idx"] == radial_idx

                attr = dict()
                attr.update(attr1)
                attr.update(attr2)
                attr["idx"] = idx
                attr["radial_idx"] = radial_idx
                attr["j"] = j_id
                attr["shape"] = (1, 1)

                yield attr
                idx += 1
            i += dim

    def steerable_attrs_j(self, j: Tuple, idx) -> Dict:
        # This attributes don't describe a single basis element but a group of basis elements which span
        # an invariant subspace. This is needed to generate the attributes of the SteerableKernelBasis.
        j_id = j
        f, j = j_id

        if f != int(j > 0):
            return

        assert idx < self.multiplicity(j_id), (idx, self.multiplicity(j_id))

        idx += sum(self.multiplicity((int(_j > 0), _j)) for _j in range(j))

        if self._steerable_idx_map is None:
            _idx = idx
        else:
            _idx = self._steerable_idx_map[idx]

        _j, radial_idx = divmod(_idx, len(self.radial))
        assert _j == j, (j, _j)

        attr1 = {"irrep:" + k: v for k, v in self.group.irrep(*j_id).attributes.items()}

        attr2 = self.radial[radial_idx]

        assert attr2["idx"] == radial_idx

        attr = dict()
        attr.update(attr1)
        attr.update(attr2)

        attr["idx"] = idx
        attr["radial_idx"] = radial_idx
        attr["j"] = j_id
        attr["shape"] = (1, 1)

        return attr

    def __getitem__(self, idx):
        assert idx < self.dim, (idx, self.dim)

        if self._idx_map is None:
            _idx = idx
        else:
            _idx = self._idx_map[idx]

        j = (_idx // len(self.radial) + 1) // 2

        assert (
            (2 * j + 1) * len(self.radial) <= _idx < (2 * j + 3) * len(self.radial)
        ), (_idx, j, self.L, len(self.radial))

        j_idx = _idx - (2 * j + 1) * len(self.radial)

        radial_idx, m = divmod(j_idx, 2 if j > 0 else 1)

        j_id = (int(j > 0), j)  # the id of the O(3) irrep
        attr1 = {"irrep:" + k: v for k, v in self.group.irrep(*j_id).attributes.items()}

        attr2 = self.radial[radial_idx]

        assert attr2["idx"] == radial_idx

        attr = dict()
        attr.update(attr1)
        attr.update(attr2)

        attr["idx"] = idx
        attr["radial_idx"] = radial_idx
        attr["j"] = j_id
        attr["m"] = m
        attr["shape"] = (1, 1)

        return attr

    def __iter__(self):
        idx = 0
        i = 0

        # since this methods return iterables of attributes built on the fly, load all attributes first and then
        # iterate on these lists
        radial_attrs = list(self.radial)

        for j in range(self.L + 1):
            dim = 2 if j > 0 else 1
            j_id = (int(j > 0), j)
            attr1 = {
                "irrep:" + k: v for k, v in self.group.irrep(*j_id).attributes.items()
            }
            for radial_idx, attr2 in enumerate(radial_attrs):
                for m in range(dim):
                    if self._filter is None or self._filter[i] == 1:
                        assert attr2["idx"] == radial_idx

                        attr = dict()
                        attr.update(attr1)
                        attr.update(attr2)
                        attr["idx"] = idx
                        attr["radial_idx"] = radial_idx
                        attr["j"] = j_id
                        attr["m"] = m
                        attr["shape"] = (1, 1)

                        yield attr
                        idx += 1
                    i += 1

    def __eq__(self, other):
        if isinstance(other, CircularShellsBasis):
            return (
                self.radial == other.radial
                and self.L == other.L
                and self._filter == other._filter
            )
        else:
            return False

    def __hash__(self):
        return self.L + hash(self.radial) + hash(self._filter)
