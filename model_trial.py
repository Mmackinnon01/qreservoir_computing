import numpy as np
import copy
import time
from sklearn.model_selection import train_test_split

from classifier.components.system import System
from classifier.components.interaction import InteractionFactory, Interaction
from classifier.components.model import Model
from classifier.components.interaction_functions import (
    CascadeFunction,
    EnergyExchangeFunction,
    DampingFunction,
)
from classifier.components.reservoir_analysis import ReservoirAnalyser

from quantum.core import GeneralQubitMatrixGen, DensityMatrix


if __name__ == "__main__":

    reservoir_nodes = [4]
    durations = [1, 2, 3, 4, 5]
    system_nodes = 2

    interfaceFactory = InteractionFactory(EnergyExchangeFunction, coupling_strength=0.5)

    reservoirFactory1 = InteractionFactory(
        EnergyExchangeFunction, coupling_strength=[0.5, 0.8]
    )

    reservoirs = []

    system_state = DensityMatrix(
        np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    )
    system_node_list = [0, 1]

    if len(system_node_list) != system_nodes:
        raise Exception

    system_interactions = {}

    system = System(
        init_quantum_state=system_state,
        nodes=system_node_list,
        interactions=system_interactions,
    )

    model = Model()
    model.setSystem(system)
    model.setReservoirInteractionFacs(
        dualFactories=[reservoirFactory1], singleFactories=[]
    )
    model.setInterfaceInteractionFacs([interfaceFactory])
    model.generateReservoir(4, init_quantum_state=0, interaction_rate=4)
    model.generateInterface(interaction_rate=2)
    model.setRunDuration(1)
    model.setRunResolution(0.1)
    model.setSwitchStructureTime(1)
    model.draw()
    for duration in durations:
        """
        Defining System setup
        """

        model.setRunDuration(duration)
        model.setSwitchStructureTime(duration)

        reservoirs.append(copy.deepcopy(model))

    analyser = ReservoirAnalyser(reservoirs)

    import pickle

    with open(
        r"C:\Users\mmack\Documents\Repos\qreservoir_computing\data\2_qubit_dataset.pkl",
        "rb",
    ) as file:
        analyser.states = pickle.load(file)

    analyser.transformStates(multiprocess=True)
