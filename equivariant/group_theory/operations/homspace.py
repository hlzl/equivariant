from typing import Tuple, Union, List

from equivariant.group_theory.groups import Group, GroupElement
from equivariant.group_theory.representations import (
    Representation,
    IrreducibleRepresentation,
)

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


__all__ = ["HomSpace"]


class HomSpace:
    def __init__(
        self,
        G: Group,
        sgid: Tuple,
    ):
        r"""
        Class defining an homogeneous space, i.e. the quotient space :math:`X \cong G / H` generated by a group
        :math:`G` and a subgroup :math:`H<G`, called the *stabilizer* subgroup.

        As a quotient space, the homogeneous space is defined as the set

        .. math::
            X \cong G / H = \{gH \ | g \in G \}

        where :math:`gH = \{gh | h \in H\}` is a *coset*.

        A classical example is given by the sphere :math:`S^2`, which can be interpreted as the quotient space
        :math:`S^2 \cong \SO3 / \SO2`, where :math:`\SO3` is the group of all 3D rotations and :math:`\SO2` here
        represents the subgroup of all planar rotations around the Z axis.

        This class is useful to generate bases for the space of functions or vector fields (Mackey functions) over the
        homogeneous space :math:`X\cong G / H`.

        Args:
            G (Group): the symmetry group of the space
            sgid (tuple): the id of the stabilizer subgroup
        """

        super(HomSpace, self).__init__()

        self.G = G
        self.H, self._inclusion, self._restriction = self.G.subgroup(sgid)
        self.sgid = sgid

        self._representations = {}
        self._names_to_irreps = {}
        self._names_to_psi = {}

    def same_coset(self, g1: GroupElement, g2: GroupElement) -> bool:
        f"""
        Check if the input elements `g1` and `g2` belong to the same coset in :math:`G/H`, i.e. if
        :math:`\exists h : g_1 = g_2 h`.
        """

        assert g1.group == self.G
        assert g2.group == self.G

        d = ~g1 @ g2

        return self._restriction(d) is not None

    def _dirac_kernel_ft(self, rho: Tuple, psi: Tuple, eps: float = 1e-9) -> np.ndarray:
        # TODO: this could be cached
        rho = self.G.irrep(*rho)
        psi = self.H.irrep(*psi)

        rho_H = rho.restrict(self.sgid)

        m_psi = 0
        for irrep in rho_H.irreps:
            if self.H.irrep(*irrep) == psi:
                m_psi += 1

        basis = np.zeros((rho.size, m_psi * psi.sum_of_squares_constituents, psi.size))

        # pick the arbitrary basis element e_i (i=0) for V_\psi
        i = 0
        p = 0
        j = 0

        column_mask = np.zeros(rho.size, dtype=bool)

        for irrep in rho_H.irreps:
            irrep = self.H.irrep(*irrep)

            if irrep == psi:
                w_i = (psi.endomorphism_basis()[:, i, :] ** 2).sum(axis=0)
                nonnull_mask = w_i > eps

                assert nonnull_mask.sum() == psi.sum_of_squares_constituents

                O_ij = np.einsum(
                    "kj,kab->ajb",
                    psi.endomorphism_basis()[:, i, nonnull_mask],
                    psi.endomorphism_basis(),
                )

                basis[
                    p : p + irrep.size, j : j + psi.sum_of_squares_constituents, :
                ] = O_ij
                column_mask[p : p + irrep.size] = nonnull_mask
                j += psi.sum_of_squares_constituents

            p += irrep.size

        if rho.sum_of_squares_constituents > 1:
            endom_basis = (
                rho_H.change_of_basis_inv[column_mask, :]
                @ rho.endomorphism_basis()
                @ rho_H.change_of_basis[:, column_mask]
            )
            ortho = (endom_basis**2).sum(0) > eps

            assert ortho.sum() == column_mask.sum() * rho.sum_of_squares_constituents, (
                ortho,
                column_mask.sum(),
                rho.sum_of_squares_constituents,
            )

            n, dependencies = connected_components(
                csgraph=csr_matrix(ortho), directed=False, return_labels=True
            )

            # check Frobenius' Reciprocity
            assert (
                n * rho.sum_of_squares_constituents
                == m_psi * psi.sum_of_squares_constituents
            ), (
                n,
                rho.sum_of_squares_constituents,
                m_psi,
                psi.sum_of_squares_constituents,
                rho,
                psi,
            )

            mask = np.zeros((ortho.shape[0]), dtype=bool)

            for i in range(n):
                columns = np.nonzero(dependencies == i)[0]
                assert len(columns) == rho.sum_of_squares_constituents
                selected_column = columns[0]
                mask[selected_column] = 1

            assert mask.sum() == n

            basis = basis[:, mask, :]

            assert basis.shape[1] == n

        basis = np.einsum("oi,ijp->ojp", rho_H.change_of_basis, basis)

        return basis

    def dimension_basis(self, rho: Tuple, psi: Tuple) -> Tuple[int, int, int]:
        r"""

        Return the tuple :math:`(\text{dim}_\rho, m, \text{dim}_\psi)`, i.e. the shape of the array returned by
        :meth:`~group.HomSpace.basis`.

        Args:
            rho (IrreducibleRepresentation): an irrep of `G` (or its id)
            psi (IrreducibleRepresentation): an irrep of `H` (or its id)

        """
        rho = self.G.irrep(*rho)
        psi = self.H.irrep(*psi)

        # Computing this restriction every time can be very expensive.
        # Representation.restrict(id) keeps a cache of the representations, so the restriction needs to be computed only
        # the first time it is called
        rho_H = rho.restrict(self.sgid)
        m_psi = rho_H.multiplicity(psi.id)

        # Frobenius' Reciprocity theorem
        multiplicity = (
            m_psi * psi.sum_of_squares_constituents / rho.sum_of_squares_constituents
        )

        assert np.isclose(multiplicity, round(multiplicity))

        multiplicity = int(round(multiplicity))

        return rho.size, multiplicity, psi.size

    def induced_representation(
        self,
        psi: Union[IrreducibleRepresentation, Tuple] = None,
        irreps: List = None,
        name: str = None,
    ) -> Representation:
        r"""
        Representation acting on the finite dimensional invariant subspace of the induced representation containing
        only the ``irreps`` passed in input.
        The induced representation is expressed in the spectral basis, i.e. as a direct sum of irreps.

        The optional parameter ``name`` is also used for caching purpose.
        Consecutive calls of this method using the same ``name`` will ignore the arguments ``psi`` and ``irreps``
        and return the same instance of representation.


        .. note::

            If ``irreps`` does not contain sufficiently many irreps, the space might be 0-dimensional.
            In this case, this method returns None.

        """
        if name is None or name not in self._representations:
            if isinstance(psi, tuple):
                psi = self.H.irrep(*psi)
            assert isinstance(psi, IrreducibleRepresentation)
            assert psi.group == self.H

            assert irreps is not None and len(irreps) > 0, irreps

            _irreps = []
            for irr in irreps:
                if isinstance(irr, tuple):
                    irr = self.G.irrep(*irr)
                assert irr.group == self.G
                _irreps.append(irr.id)
            irreps = _irreps

            # check there are no duplicates
            assert len(irreps) == len(set(irreps)), irreps

        if name is None:
            irreps_names = "|".join(str(i) for i in irreps)
            name = f"induced[{self.sgid}]_[{psi.id}]_[{irreps_names}]"

        if name not in self._representations:
            assert irreps is not None and len(irreps) > 0, irreps

            irreps_ids = []
            size = 0
            for irr in irreps:
                irr_size, multiplicity = self.dimension_basis(irr, psi.id)[:2]
                irreps_ids += [irr] * multiplicity

                size += multiplicity * irr_size

            if size == 0:
                return None

            self._names_to_irreps[name] = irreps
            self._names_to_psi[name] = psi.id

            supported_nonlinearities = ["norm", "gated", "concatenated"]
            self._representations[name] = Representation(
                self.G,
                name,
                irreps_ids,
                change_of_basis=np.eye(size),
                supported_nonlinearities=supported_nonlinearities,
            )

        return self._representations[name]