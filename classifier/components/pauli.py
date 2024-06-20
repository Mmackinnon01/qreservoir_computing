import numpy as np
from itertools import product
from quantum.core import Operator
from quantum.operator import sigmaI, sigmaX, sigmaY, sigmaZ

sigma_plus = Operator(np.array([[0, 1], [0, 0]]))
sigma_minus = Operator(np.array([[0, 0], [1, 0]]))


def compositeOperator(operator, operation_nodes, n_nodes):
    operators = [sigmaI for i in range(n_nodes)]

    for node in operation_nodes:
        operators[node] = operator

    total_state = operators[0]

    for i in range(n_nodes - 1):
        total_state = total_state.tensor(operators[i + 1])

    return total_state


def compositeSigmaPlus(node, n_nodes):
    return compositeOperator(sigma_plus, [node], n_nodes)


def compositeSigmaMinus(node, n_nodes):
    return compositeOperator(sigma_minus, [node], n_nodes)


def compositeSigmaX(node, n_nodes):
    return compositeOperator(sigmaX, [node], n_nodes)


def compositeSigmaY(node, n_nodes):
    return compositeOperator(sigmaY, [node], n_nodes)


def compositeSigmaZ(node, n_nodes):
    return compositeOperator(sigmaZ, [node], n_nodes)


def compositeSigmaXMulti(nodes, n_nodes):
    return compositeOperator(sigmaX, nodes, n_nodes)


def compositeSigmaYMulti(nodes, n_nodes):
    return compositeOperator(sigmaY, nodes, n_nodes)


def sigmaCombinatorics(n_nodes):
    operators = {}
    for combination in list(
        product([["x", sigmaX], ["y", sigmaY], ["z", sigmaZ]], repeat=n_nodes)
    ):
        name = ""
        first = True
        for sigma in combination:
            name += sigma[0]
            if first:
                first = False
                operator = sigma[1]
            else:
                operator = operator.tensor(sigma[1])
        operators[name] = operator
    return operators


print(compositeSigmaXMulti([0, 1], 2) + compositeSigmaYMulti([0, 1], 2))
