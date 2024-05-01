class System:
    def __init__(self, init_quantum_state, nodes: list, interactions: dict):
        self.init_quantum_state = init_quantum_state
        self.nodes = nodes
        self.interactions = interactions

    def calcDensityDerivative(self, model_state, structure_phase):
        density_derivative = 0

        for interaction in self.interactions.values():
            if type(interaction) == list:
                interaction = interaction[structure_phase]
            if not interaction.function.unitary():
                density_derivative += interaction.calc(model_state)
        return density_derivative

    def calcUnitaryH(self, structure_phase):
        H = 0

        for interaction in self.interactions.values():
            if type(interaction) == list:
                interaction = interaction[structure_phase]
            if interaction.function.unitary():
                H += interaction.function.H
        self.unitary_H = H
