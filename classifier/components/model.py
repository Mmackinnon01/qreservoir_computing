from .reservoir import Reservoir
from .interface import Interface
from .model_log import ModelLog
from .runge_kutta import rungeKutta, rungeKuttaG
from .measure_excitations import (
    measureAllExcitations,
    measureTotalExcitations,
    measureAllSigmaCombinations,
)
import numpy as np
import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import math

sns.set()
sns.set_style("darkgrid")
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"] = 1

from quantum.core import DensityMatrix


class Model:
    def __init__(self):
        self.switch_structure_time = 99999
        self.runge_kutta_a = [
            [0, 0, 0, 0],
            [1 / 2, 0, 0, 0],
            [0, 1 / 2, 0, 0],
            [0, 0, 1, 0],
        ]
        self.runge_kutta_b = [1 / 6, 1 / 3, 1 / 3, 1 / 6]

    def __repr__(self):
        return f"Quantum Reservoir with {len(self.reservoir.nodes)} nodes, {len(self.reservoir.dualInteractions)} internal connections and {len(self.interface.interactions)} interface connections"

    def draw(self, ax=None):
        # Step 1: Create a graph object
        G = nx.Graph()

        # Step 2: Add nodes and edges
        G.add_nodes_from(np.arange(len(self.system.nodes) + len(self.reservoir.nodes)))

        # Add edges
        interactions = []
        edge_styles = []
        edge_colours = []
        edge_labels = {}

        for interaction in self.interface.interactions.keys():
            interactions.append((int(interaction[-2]), int(interaction[-1])))
            edge_styles.append(
                (int(interaction[-2]), int(interaction[-1]), {"style": "dotted"})
            )
            label = ""

            if type(self.interface.interactions[interaction]) == list:
                for fac in self.interface.interactions[interaction]:
                    for i, val in enumerate(fac.variables.values()):
                        label += str(round(val, 2))

                        if i != len(fac.variables) - 1:
                            label += " / "
                    label += " -> "
                label = label[:-4]
            else:
                for i, val in enumerate(
                    self.interface.interactions[interaction].variables.values()
                ):
                    label += str(round(val, 2))

                    if i != len(self.interface.interactions[interaction].variables) - 1:
                        label += " / "
            edge_labels[(int(interaction[-2]), int(interaction[-1]))] = label
            edge_colours.append("red")

        for interaction in self.reservoir.dualInteractions.keys():
            interactions.append((int(interaction[-2]), int(interaction[-1])))
            edge_styles.append(
                (int(interaction[-2]), int(interaction[-1]), {"style": "solid"})
            )
            label = ""

            for i, val in enumerate(
                self.reservoir.dualInteractions[interaction].variables.values()
            ):
                label += str(round(val, 2))

                if i != len(self.reservoir.dualInteractions[interaction].variables) - 1:
                    label += " / "
            edge_labels[(int(interaction[-2]), int(interaction[-1]))] = label
            edge_colours.append("black")

        G.add_edges_from(interactions)

        # Step 3: Define positions for the nodes
        positions = {}

        for node in range(len(self.system.nodes)):
            positions[node] = (0, 0.5 - (1 / (len(self.system.nodes) - 1)) * node)

        for node in range(
            len(self.system.nodes), len(self.system.nodes) + len(self.reservoir.nodes)
        ):
            positions[node] = (
                2 + math.sin(((node / len(self.reservoir.nodes)) + 0.2) * 2 * math.pi),
                math.cos(((node / len(self.reservoir.nodes)) + 0.2) * 2 * math.pi),
            )
        # Step 4: Draw the graph

        {("A", "B"): "AB", ("B", "C"): "BC", ("B", "D"): "BD"}

        if not ax:
            plot = True
            fig = plt.figure()
            ax = fig.subplots()
        else:
            plot = False

        nx.draw(
            G,
            positions,
            ax=ax,
            edgelist=edge_styles,
            edge_color=edge_colours,
            with_labels=True,
            node_color="skyblue",
            node_size=1000,
            font_size=12,
            font_weight="bold",
        )
        nx.draw_networkx_edges(
            G, positions, ax=ax, edgelist=edge_styles, edge_color=edge_colours
        )

        nx.draw_networkx_edge_labels(
            G,
            positions,
            ax=ax,
            edge_labels=edge_labels,
            font_color="blue",
            bbox=dict(alpha=0),
        )

        ax.vlines([0.5], ymin=-2, ymax=2)
        ax.text(-0.15, 2, "System")
        ax.text(1.85, 2, "Reservoir")

        if plot:
            plt.show()

    def setSystem(self, system):
        self.system = system

    def setReservoirInteractionFacs(self, dualFactories, singleFactories):
        self.reservoirDualInteractionFacs = dualFactories
        self.reservoirSingleInteractionFacs = singleFactories

    def setInterfaceInteractionFacs(self, factories):
        self.interfaceInteractionFacs = factories

    def generateReservoir(self, n_nodes, init_quantum_state=0, interaction_rate=1):
        self.reservoir = Reservoir(
            self.reservoirSingleInteractionFacs, self.reservoirDualInteractionFacs
        )
        self.reservoir.setupNodes(
            n_nodes=n_nodes,
            system_nodes=len(self.system.nodes),
            quantum_state=init_quantum_state,
        )
        self.reservoir.computeInitialQuantumState()
        self.reservoir.setupSingleInteractions()
        self.reservoir.setupDualInteractions(interaction_rate)

    def generateInterface(self, interaction_rate=1):
        self.interface = Interface(
            sys_nodes=self.system.nodes,
            res_nodes=list(self.reservoir.nodes.keys()),
            interactionFactories=self.interfaceInteractionFacs,
        )
        self.interface.setupInteractions(interaction_rate)

    def setRunDuration(self, run_duration):
        self.run_duration = run_duration

    def setRunResolution(self, run_resolution):
        self.run_timestep = run_resolution

    def setSwitchStructureTime(self, switch_structure_time):
        self.switch_structure_time = switch_structure_time

    def calcIterations(self):
        self.iterations = round(self.run_duration / self.run_timestep)

    def calcUnitaryH(self):
        self.system.calcUnitaryH(self.structure_phase)
        self.reservoir.calcUnitaryH(self.structure_phase)
        self.interface.calcUnitaryH(self.structure_phase)

        return (
            self.system.unitary_H + self.reservoir.unitary_H + self.interface.unitary_H
        )

    def calcStartingState(self):
        self.current_state = self.system.init_quantum_state.tensor(
            self.reservoir.init_quantum_state
        )

        self.unitary_H = self.calcUnitaryH()
        self.calcTraceState()
        self.calcExcitationState()

    def run(self, measure=True):
        self.structure_phase = 0
        self.setupModelLog()
        self.calcIterations()
        self.calcStartingState()
        self.logIteration()

        for step in range(self.iterations):
            self.updateState(measure)
            self.logIteration(measure)
            if round(self.run_timestep * step, 3) == self.switch_structure_time:
                self.switchStructure()

    def switchStructure(self):
        self.structure_phase = 1
        self.calcUnitaryH()

    def logIteration(self, measure=True):
        self.modelLog.addLogEntry(self.current_state)
        self.modelLog.addTraceLogEntry(self.current_trace_state)
        if measure:
            self.modelLog.addExcitationLogEntry(self.current_excitation_expectations)
            self.modelLog.addTotalExcitationLogEntry(
                self.current_total_excitation_expectations
            )
            self.modelLog.addSigmaCombinationLogEntry(
                self.current_sigma_combination_expectations
            )
        self.modelLog.moveTimeStep()

    def setupModelLog(self):
        self.modelLog = ModelLog(self.run_timestep)

    def updateState(self, measure=True):
        self.current_state = rungeKuttaG(
            self.calcDensityDerivative,
            self.run_timestep,
            self.current_state,
            a=self.runge_kutta_a,
            b=self.runge_kutta_b,
        )
        self.calcTraceState()
        if measure:
            self.calcExcitationState()

    def calcExcitationState(self):
        self.current_excitation_expectations = measureAllExcitations(
            self.current_trace_state
        )
        self.current_total_excitation_expectations = measureTotalExcitations(
            self.current_trace_state
        )
        self.current_sigma_combination_expectations = measureAllSigmaCombinations(
            self.current_trace_state
        )

    def calcTraceState(self):
        self.current_trace_state = self.trace(
            self.current_state, len(self.system.nodes)
        )

    def trace(self, density_matrix: DensityMatrix, system_nodes):
        for i in range(system_nodes):
            density_matrix = density_matrix.partialTrace(0)
        return density_matrix

    def calcDensityDerivative(self, state):
        system_component = self.system.calcDensityDerivative(
            state, self.structure_phase
        )
        reservoir_component = self.reservoir.calcDensityDerivative(
            state, self.structure_phase
        )
        interface_component = self.interface.calcDensityDerivative(
            state, self.structure_phase
        )
        return (
            reservoir_component
            + system_component
            + interface_component
            + self.calcUnitaryComponent(state)
        )

    def calcUnitaryComponent(self, ro):
        value = -1j * self.unitary_H.commutator(ro)
        return value

    def transform(self, starting_state):
        self.system.init_quantum_state = starting_state
        self.run(measure=False)
        self.calcExcitationState()
        return np.real(list(self.current_sigma_combination_expectations.values()))
