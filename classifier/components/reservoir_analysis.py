from .model import Model
from multiprocessing import pool

from quantum.core import GeneralQubitMatrixGen

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

import numpy as np
import tqdm
import pickle
import os
import datetime
import matplotlib.pyplot as plt


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
            return self.state_gen.generatePartiallyEntangledState(
                n_qubits=n_qubits, degree=n_qubits - 1
            )

    def generateStates(self, nstates, n_qubits=1, state_subset="general"):
        self.states = [
            self.generateState(n_qubits, state_subset) for i in range(nstates)
        ]

    def transformStates(self, multiprocess=False):
        for i, reservoir in enumerate(self.reservoirs):
            print(f"Transforming reservoir {i+1} of {len(self.reservoirs)}")

            if multiprocess:
                with pool.Pool() as p:
                    d_train = p.map(
                        reservoir.transform, tqdm.tqdm(self.states), chunksize=100
                    )
            else:

                d_train = [
                    reservoir.transform(state) for state in tqdm.tqdm(self.states)
                ]

            dataset = ReservoirAnalysisDataset(self.states, d_train, reservoir)
            dataset.train()
            dataset.dump()
            self.datasets.append(dataset)

    def compare(self):
        fig = plt.figure(figsize=(14, 5 * len(self.datasets)))
        axs = fig.subplots(nrows=len(self.datasets), ncols=2)

        for i, dataset in enumerate(self.datasets):
            model_ax = axs[i][0]
            data_ax = axs[i][1]

            dataset.reservoir.draw(model_ax)
            data_ax.scatter(
                dataset.y_test,
                [
                    0 if pred < 0 else pred
                    for pred in dataset.mlp.predict(dataset.x_test)
                ],
            )
            data_ax.set_xlabel("Target Negativity")
            data_ax.set_ylabel("Approximation Negativity")
            data_ax.set_xlim(-0.1, 1.1)
            data_ax.set_ylim(-0.1, 1.1)
            data_ax.set_title(f"MLP Score of {dataset.score}")

        fig.suptitle(
            "Comparison of Negativity Reconstruction Accuracies", fontsize=15, y=0.95
        )
        plt.show()


class ReservoirAnalysisDataset:

    def __init__(self, target_states, transformed_states, reservoir):
        self.target_states = target_states
        self.transformed_states = transformed_states
        self.reservoir = reservoir

        self.nstates = len(self.target_states)
        self.n_qubits = int(np.log2(self.target_states[0].matrix.shape[0]))

    def train(self):
        y = [state.negativity() for state in self.target_states]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.transformed_states, y, test_size=0.1
        )
        self.mlp = MLPRegressor().fit(np.real(self.x_train), self.y_train)
        self.score = self.mlp.score(np.real(self.x_test), self.y_test)

    def dump(self):
        now = datetime.datetime.now()

        with open(
            os.path.join(
                os.getcwd(), f'qrc_dataset_{2}_qubit_{now.strftime("%d%m%Y%H%M%S")}.pkl'
            ),
            "wb",
        ) as file:
            pickle.dump(self, file)
