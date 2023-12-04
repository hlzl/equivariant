from __future__ import annotations

from .group import Group, GroupElement
from equivariant.group_theory import *

from .utils import *

import numpy as np
import math
from functools import partial

from typing import Tuple, Callable, Iterable, List, Dict, Any


__all__ = ["CyclicGroup"]


class CyclicGroup(Group):
    def __init__(self, N: int):
        r"""
        Build an instance of the cyclic group :math:`C_N` which contains :math:`N` discrete planar rotations.

        The group elements are :math:`\{e, r, r^2, r^3, \dots, r^{N-1}\}`, with group law
        :math:`r^a \cdot r^b = r^{\ a + b \!\! \mod \!\! N \ }`.
        The cyclic group :math:`C_N` is isomorphic to the integers *modulo* ``N``.
        However, elements are stored as radians between :math:`0` and :math:`2*pi`, where the :math:`k`-th
        element can also be interpreted as the discrete rotation by :math:`k\frac{2\pi}{N}`.

        Subgroup Structure.

        A subgroup of :math:`C_N` is another cyclic group :math:`C_M` and is identified by an ``id`` containing the
        integer :math:`M` (i.e. the order of the subgroup).

        If the current group is :math:`C_N`, the subgroup is generated by :math:`r^{(N/M)}`.
        Notice that :math:`M` has to divide the order :math:`N` of the group.

        Args:
            N (int): order of the group

        """

        assert isinstance(N, int) and N > 0, N

        super(CyclicGroup, self).__init__("C%d" % N, False, True)

        self.rotation_order = N
        self._elements = [self.element(i * (2 * np.pi) / N) for i in range(N)]
        self._identity = self.element(0.0)

        self._build_representations()

    @property
    def generators(self) -> List[GroupElement]:
        if self.rotation_order > 1:
            return [self.element(1)]
        else:
            # the generator of the trivial group is the empty set
            return []

    @property
    def subgroup_trivial_id(self):
        return 1

    @property
    def subgroup_self_id(self):
        return self.order()

    ###########################################################################
    # METHODS DEFINING THE GROUP LAW AND THE OPERATIONS ON THE GROUP'S ELEMENTS
    ###########################################################################

    def sample(self) -> GroupElement:
        return self.element(
            np.random.randint(0, self.rotation_order)
            * (2 * np.pi)
            / self.rotation_order
        )

    def testing_elements(self) -> Iterable[GroupElement]:
        r"""
        A finite number of group elements to use for testing.

        """
        return iter(self._elements)

    def __eq__(self, other):
        if not isinstance(other, CyclicGroup):
            return False
        else:
            return self.name == other.name and self.order() == other.order()

    def _subgroup(
        self, id: int
    ) -> Tuple[
        Group,
        Callable[[GroupElement], GroupElement],
        Callable[[GroupElement], GroupElement],
    ]:
        r"""
        Restrict the current group to the cyclic subgroup :math:`C_M`.
        If the current group is :math:`C_N`, it restricts to the subgroup generated by :math:`r^{(N/M)}`.
        Notice that :math:`M` has to divide the order :math:`N` of the current group.

        The method takes as input the integer :math:`M` identifying of the subgroup to build (the order of the subgroup)

        Args:
            id (int): the integer :math:`M` identifying of the subgroup

        Returns:
            a tuple containing
                - the subgroup,
                - a function which maps an element of the subgroup to its inclusion in the original group and
                - a function which maps an element of the original group to the corresponding element in the subgroup
                  (returns None if the element is not contained in the subgroup)
        """

        assert isinstance(id, int), id

        order = id

        assert self.order() % order == 0, (
            "Error! The subgroups of a cyclic group have an order that divides the order of the supergroup."
            " %d does not divide %d " % (order, self.order())
        )

        # Build the subgroup
        # take the elements of the group generated by "r^ratio"
        sg = CyclicGroup(order)
        parent_mapping = partial(_build_parent_map, self, order)
        child_mapping = partial(_build_child_map, self, sg)

        return sg, parent_mapping, child_mapping

    def grid(self, type: str, N: int) -> List[GroupElement]:
        r"""
        .. todo ::
            Add docs

        """
        if type == "rand":
            return [self.sample() for _ in range(N)]
        elif type == "regular":
            assert self.order() % N == 0
            r = self.order() // N
            return [self.element(i * r) for i in range(N)]
        else:
            raise ValueError(f'Grid type "{type}" not recognized!')

    def _combine_subgroups(self, sg_id1, sg_id2):
        sg_id1 = self._process_subgroup_id(sg_id1)
        sg1, _, _ = self.subgroup(sg_id1)
        sg_id2 = sg1._process_subgroup_id(sg_id2)

        return sg_id2

    def _build_representations(self):
        r"""
        Build the irreps and the regular representation for this group

        """

        N = self.order()

        # Build all the irreducible representations
        for k in range(0, int(N // 2) + 1):
            self.irrep(k)

        # Build all group_theory.Representations
        # add all the irreps to the set of representations already built for this group
        self.representations.update(**{irr.name: irr for irr in self.irreps()})

        # build the regular representation
        self.representations["regular"] = self.regular_representation
        self.representations["regular"].supported_nonlinearities.add("vectorfield")

    def _build_quotient_representations(self):
        r"""
        Build all the quotient representations for this group

        """
        for n in range(2, int(math.ceil(math.sqrt(self.order())))):
            if self.order() % n == 0:
                self.quotient_representation(n)

    @property
    def trivial_representation(self) -> group_theory.Representation:
        return self.representations["irrep_0"]

    def irrep(self, k: int) -> group_theory.IrreducibleRepresentation:
        r"""
        Build the irrep of frequency ``k`` of the current cyclic group.
        The frequency has to be a non-negative integer in :math:`\{0, \dots, \left \lfloor N/2 \right \rfloor \}`,
        where :math:`N` is the order of the group.

        Args:
            k (int): the frequency of the representation

        Returns:
            the corresponding irrep

        """
        id = (k,)

        if id not in self._irreps:
            assert 0 <= k <= self.order() // 2, (k, self.order())
            name = f"irrep_{k}"

            n = self.order()
            if k == 0:
                # Trivial representation
                irrep = partial(_build_irrep_cn, 0)
                character = partial(_build_char_cn, 0)
                supported_nonlinearities = [
                    "pointwise",
                    "gate",
                    "norm",
                    "gated",
                    "concatenated",
                ]
                self._irreps[id] = group_theory.IrreducibleRepresentation(
                    self,
                    id,
                    name,
                    irrep,
                    1,
                    "R",
                    supported_nonlinearities=supported_nonlinearities,
                    character=character,
                    frequency=k,
                )
            elif n % 2 == 0 and k == int(n / 2):
                # 1 dimensional Irreducible representation (only for even order groups)
                irrep = partial(_build_irrep_cn, k)
                character = partial(_build_char_cn, k)
                supported_nonlinearities = ["norm", "gated", "concatenated"]
                self._irreps[id] = group_theory.IrreducibleRepresentation(
                    self,
                    id,
                    name,
                    irrep,
                    1,
                    "R",
                    supported_nonlinearities=supported_nonlinearities,
                    character=character,
                    frequency=k,
                )
            else:
                # 2 dimensional Irreducible group_theory.Representations
                irrep = partial(_build_irrep_cn, k)
                character = partial(_build_char_cn, k)

                supported_nonlinearities = ["norm", "gated"]
                self._irreps[id] = group_theory.IrreducibleRepresentation(
                    self,
                    id,
                    name,
                    irrep,
                    2,
                    "C",
                    supported_nonlinearities=supported_nonlinearities,
                    character=character,
                    frequency=k,
                )
        return self._irreps[id]

    def clebsch_gordan_coeff(self, m, n, j) -> np.ndarray:
        (m,) = self.get_irrep_id(m)
        (n,) = self.get_irrep_id(n)
        (j,) = self.get_irrep_id(j)

        rho_m = self.irrep(m)
        rho_n = self.irrep(n)
        rho_j = self.irrep(j)

        if m == 0 or n == 0:
            if j == m + n:
                return np.eye(rho_j.size).reshape(rho_m.size, rho_n.size, 1, rho_j.size)
            else:
                return np.zeros((rho_m.size, rho_n.size, 0, rho_j.size))
        elif (self.order() % 2 == 0) and (
            m == self.order() // 2 or n == self.order() // 2
        ):
            if j == m + n:
                return np.eye(rho_j.size).reshape(rho_m.size, rho_n.size, 1, rho_j.size)
            elif j == (self.order() - m - n):
                cg = np.eye(rho_j.size)
                if rho_j.size > 1:
                    cg[:, 1] *= -1
                return cg.reshape(rho_m.size, rho_n.size, 1, rho_j.size)
            else:
                return np.zeros((rho_m.size, rho_n.size, 0, rho_j.size))
        else:
            cg = np.array(
                [
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [0.0, -1.0, 0.0, 1.0],
                    [1.0, 0.0, -1.0, 0.0],
                ]
            ) / np.sqrt(2)
            if j == m + n:
                cg = cg[:, 2:]
            elif j == self.order() - m - n:
                cg = cg[:, 2:]
                cg[:, 1] *= -1
            elif j == m - n:
                cg = cg[:, :2]
            elif j == n - m:
                cg = cg[:, :2]
                cg[:, 1] *= -1
            else:
                cg = np.zeros((rho_m.size, rho_n.size, 0, rho_j.size))

            return cg.reshape(rho_n.size, rho_m.size, -1, rho_j.size).transpose(
                1, 0, 2, 3
            )

    def _tensor_product_irreps(self, J: int, l: int) -> List[Tuple[Tuple, int]]:
        (J,) = self.get_irrep_id(J)
        (l,) = self.get_irrep_id(l)

        if J == 0 or l == 0:
            return [((l + J,), 1)]
        elif (self.order() % 2 == 0) and (
            J == self.order() // 2 or l == self.order() // 2
        ):
            j = (J + l) if (J + l <= self.order() // 2) else (self.order() - J - l)
            return [((j,), 1)]
        elif l == J:
            j = (J + l) if (J + l <= self.order() // 2) else (self.order() - J - l)
            m = 1 if j < self.order() / 2 else 2
            return [
                ((0,), 2),
                ((j,), m),
            ]
        else:
            j = (J + l) if (J + l <= self.order() // 2) else (self.order() - J - l)
            m = 1 if j < self.order() / 2 else 2
            return [
                ((np.abs(l - J),), 1),
                ((j,), m),
            ]


def _build_irrep_cn(k: int, element: GroupElement) -> np.ndarray:
    if k == 0:
        return np.eye(1)

    n = element.group.order()

    if n % 2 == 0 and k == int(n / 2):
        # 1 dimensional Irreducible representation (only for even order groups)
        return np.array([[np.cos(k * element.value)]])
    else:
        # 2 dimensional Irreducible group_theory.Representations
        return utils.psi(element.value, k=k)


def _build_char_cn(k: int, element: GroupElement) -> float:
    if k == 0:
        return 1.0

    n = element.group.order()

    if n % 2 == 0 and k == int(n / 2):
        # 1 dimensional irreducible representation (only for even order groups)
        return np.cos(k * element.value)
    else:
        # 2 dimensional irreducible representations
        return 2 * np.cos(k * element.value)


def _build_parent_map(G: CyclicGroup, order: int, e: GroupElement) -> GroupElement:
    return G.element(e * G.order() // order)


def _build_child_map(G: CyclicGroup, sg: CyclicGroup, e: GroupElement) -> GroupElement:
    assert G.order() % sg.order() == 0
    assert e.group == G
    i = e.value
    ratio = G.order() // sg.order()
    if i % ratio != 0:
        return None
    else:
        return sg.element(i // ratio)