from .model import Model
from multiprocessing import pool

from quantum.core import GeneralQubitMatrixGen

import copy
import numpy as np
import tqdm


class ReservoirAnalyser:

    def __init__(self, reservoirs):
        self.state_gen = GeneralQubitMatrixGen()
        self.reservoirs = reservoirs
        self.datasets = []


    def generateState(self, n_qubits, state_subset):
        if state_subset == "general":
            return self.state_gen.generateState(n_qubits=n_qubits)
        elif state_subset == "separable":
            return self.state_gen.generateSeparableState(n_qubits=n_qubits)
        elif state_subset == "mixed":
            return self.state_gen.generateMixedState(n_qubits=n_qubits)
        elif state_subset == "pure":
            return self.state_gen.generatePureState(n_qubits=n_qubits)
        elif state_subset == "werner":
            return self.state_gen.generateWernerState(c=np.random.rand())
        elif state_subset == "partial_entangle":
            return self.state_gen.generatePartiallyEntangledState(n_qubits=n_qubits, degree=n_qubits-1)

    def generateStates(self, nstates, n_qubits = 1, state_subset="general"):
        self.states = [self.generateState(n_qubits, state_subset) for i in range(nstates)]

    def transformStates(self, multiprocess=False):
        for i, reservoir in enumerate(self.reservoirs):
            print(f'Transforming reservoir {i+1}')
            
            if multiprocess:
                with pool.Pool() as p:
                    d_train = p.map(reservoir.transform, tqdm.tqdm(self.states), chunksize=10)

            else:

                d_train = [reservoir.transform(state) for state in tqdm.tqdm(self.states)]

            dataset = ReservoirAnalysisDataset(self.states, d_train, reservoir)
            self.datasets.append(dataset)
    
class ReservoirAnalysisDataset:

    def __init__(self, target_states, transformed_states, reservoir):
        self.target_states = target_states
        self.transformed_states = transformed_states
        self.reservoir = reservoir

        self.nstates = len(self.target_states)
        self.n_qubits = int(np.log2(self.target_states[0].matrix.shape[0]))
