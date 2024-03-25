import numpy as np
from sklearn.model_selection import train_test_split

from classifier.components.system import System
from classifier.components.interaction import InteractionFactory, Interaction
from classifier.components.model import Model
from classifier.components.interaction_functions import CascadeFunction, EnergyExchangeFunction, DampingFunction
from classifier.components.reservoir_analysis import ReservoirAnalyser

from quantum.core import GeneralQubitMatrixGen, DensityMatrix

reservoir_nodes=[5]
system_nodes=2


interfaceFactory = InteractionFactory(CascadeFunction, gamma_1=1, gamma_2=1)

reservoirFactory1 = InteractionFactory(EnergyExchangeFunction, coupling_strength=1)
reservoirFactory2 = InteractionFactory(DampingFunction, damping_strength=1)

reservoirs = []

for n_reservoir_nodes in reservoir_nodes:
    """
    Defining System setup
    """

    system_state = DensityMatrix(np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
    system_node_list = [0, 1]

    if len(system_node_list) != system_nodes:
        raise Exception

    system_interactions = {"sys_interaction_0" : Interaction(0, DampingFunction(0, n_reservoir_nodes+system_nodes, 1)),
                        "sys_interaction_1" : Interaction(1, DampingFunction(1, n_reservoir_nodes+system_nodes, 1))}

    system = System(
        init_quantum_state=system_state, nodes=system_node_list, interactions=system_interactions
    )



    model = Model()
    model.setSystem(system)
    model.setReservoirInteractionFacs(dualFactories=[reservoirFactory1], singleFactories=[reservoirFactory2])
    model.setInterfaceInteractionFacs([interfaceFactory])
    model.generateReservoir(n_reservoir_nodes, init_quantum_state=0, interaction_rate=.5)
    model.generateInterface(interaction_rate=.2)
    model.setRunDuration(1)
    model.setRunResolution(0.2)
    model.setSwitchStructureTime(2)

    reservoirs.append(model)

reservoirs[0].draw()

analyser = ReservoirAnalyser(reservoirs)

import pickle

with open(f"/Users/matthewmackinnon/Documents/repos/entanglement_classifier_redesign/data/2_qubit_dataset.pkl", 'rb') as file:
    analyser.states = pickle.load(file)

if __name__ == "__main__":

    analyser.transformStates(multiprocess=False)

with open(f'/Users/matthewmackinnon/Documents/repos/entanglement_classifier_redesign/data/5_qubit_reservoir.pkl', 'wb') as file:
        pickle.dump(analyser.datasets[0], file)