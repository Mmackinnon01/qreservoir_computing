import numpy as np

from quantum.core import DensityMatrix
from .pauli import compositeSigmaZ, compositeSigmaXMulti, compositeSigmaYMulti, sigmaCombinatorics


def measureAllExcitations(density_matrix):
    excitations = []
    for system in np.arange(density_matrix.n_systems):
        excitation_val = measureExcitation(density_matrix, system)
        excitations.append(excitation_val)
    return excitations


def measureExcitation(density_matrix: DensityMatrix, system):
    operator = compositeSigmaZ(
        system, density_matrix.n_systems)
    return density_matrix.measure(operator)


def measureTotalExcitations(density_matrix: DensityMatrix):
    n_nodes = density_matrix.n_systems
    x_operator = compositeSigmaXMulti([node for node in range(n_nodes)], n_nodes)
    y_operator = compositeSigmaYMulti([node for node in range(n_nodes)], n_nodes)
    return density_matrix.measure(x_operator), density_matrix.measure(y_operator)


def measureAllSigmaCombinations(density_matrix: DensityMatrix):
    n_nodes = density_matrix.n_systems
    operators = sigmaCombinatorics(n_nodes)
    for name, operator in operators.items():
        operators[name] = density_matrix.measure(operator)
    return operators
