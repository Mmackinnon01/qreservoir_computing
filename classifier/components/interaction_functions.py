import numpy as np

from .pauli import (
    compositeSigmaPlus,
    compositeSigmaMinus,
    compositeSigmaXMulti,
    compositeSigmaYMulti,
)
from quantum.core import DensityMatrix


class UnitaryFunction:

    def __init__(self):
        self.unitary = True


class NonUnitaryFunction:

    def __init__(self):
        self.unitary = False


class DampingFunction(NonUnitaryFunction):
    def __init__(self, nodes, n_nodes, damping_strength):
        super().__init__()
        self.node = nodes
        self.n_nodes = n_nodes
        self.damping_strength = damping_strength
        self.sigma_plus = compositeSigmaPlus(self.node, n_nodes)
        self.sigma_minus = compositeSigmaMinus(self.node, n_nodes)
        self.sigma_plus_minus = self.sigma_plus * self.sigma_minus

    def calc(self, ro: DensityMatrix):
        value = (self.damping_strength / 2) * (
            2 * self.sigma_minus * ro * self.sigma_plus
            - ro * self.sigma_plus_minus
            - self.sigma_plus_minus * ro
        )
        return value


class CascadeFunction(NonUnitaryFunction):
    def __init__(self, nodes, n_nodes, gamma_1, gamma_2):
        super().__init__()
        self.node1 = nodes[0]
        self.node2 = nodes[1]
        self.n_nodes = n_nodes
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.sigma_plus_1 = compositeSigmaPlus(self.node1, n_nodes)
        self.sigma_plus_2 = compositeSigmaPlus(self.node2, n_nodes)
        self.sigma_minus_1 = compositeSigmaMinus(self.node1, n_nodes)
        self.sigma_minus_2 = compositeSigmaMinus(self.node2, n_nodes)

    def calc(self, ro):
        term1 = self.sigma_plus_2.commutator(self.sigma_minus_1 * ro)
        term2 = (ro * self.sigma_plus_1).commutator(self.sigma_minus_2)
        return -((self.gamma_1 * self.gamma_2) ** 0.5) * (term1 + term2)


class EnergyExchangeFunction(UnitaryFunction):
    def __init__(self, nodes, n_nodes, coupling_strength):
        super().__init__()
        self.node1 = nodes[0]
        self.node2 = nodes[1]
        self.n_nodes = n_nodes
        self.coupling_strength = coupling_strength
        self.sigma_x_12 = compositeSigmaXMulti([self.node1, self.node2], n_nodes)
        self.sigma_y_12 = compositeSigmaYMulti([self.node1, self.node2], n_nodes)
        self.H = self.coupling_strength * (self.sigma_x_12 + self.sigma_y_12)

    def calc(self, ro: DensityMatrix):

        value = -1j * self.H.commutator(ro)
        return value


class DampedCascadeFunction(NonUnitaryFunction):
    def __init__(self, nodes, n_nodes, gamma_1, gamma_2):
        super().__init__()
        self.node1 = nodes[0]
        self.node2 = nodes[1]
        self.n_nodes = n_nodes
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.sigma_plus_1 = compositeSigmaPlus(self.node1, n_nodes)
        self.sigma_plus_2 = compositeSigmaPlus(self.node2, n_nodes)
        self.sigma_minus_1 = compositeSigmaMinus(self.node1, n_nodes)
        self.sigma_minus_2 = compositeSigmaMinus(self.node2, n_nodes)

    def calc(self, ro: DensityMatrix):

        term1 = (self.gamma_1) * (
            2 * self.sigma_minus_1 * ro * self.sigma_plus_1
            - ro * self.sigma_plus_1 * self.sigma_minus_1
            - self.sigma_plus_1 * self.sigma_minus_1 * ro
        )
        term2 = (
            0.1
            * (self.gamma_2)
            * (
                2 * self.sigma_minus_2 * ro * self.sigma_plus_2
                - ro * self.sigma_plus_2 * self.sigma_minus_2
                - self.sigma_plus_2 * self.sigma_minus_2 * ro
            )
        )
        term3 = -((self.gamma_1 * self.gamma_2) ** 0.5) * self.sigma_plus_2.commutator(
            self.sigma_minus_1 * model_state
        )
        term4 = -((self.gamma_1 * self.gamma_2) ** 0.5) * (
            model_state * self.sigma_plus_1
        ).commutator(self.sigma_minus_2)
        return term1 + term2 + term3 + term4
