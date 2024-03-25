import matplotlib.pyplot as plt
import numpy as np

from quantum.core import DensityMatrix

from .plots import plotReservoir, plotReservoirAndSystem, plotExcitations, plotTotalExcitations, plotSigmaCombinations


class ModelLog:
    def __init__(self, run_resolution):
        self.time_log = {}
        self.partial_time_log = {}
        self.excitation_time_log = {}
        self.total_excitation_time_log = {}
        self.sigma_combination_time_log = {}
        self.current_time = 0
        self.run_resolution = run_resolution

    def addLogEntry(self, entry):
        self.evalulateEntry(entry)
        self.time_log[self.current_time] = entry

    def addTraceLogEntry(self, entry):
        self.evalulateEntry(entry)
        self.partial_time_log[self.current_time] = entry

    def addExcitationLogEntry(self, entry):
        self.excitation_time_log[self.current_time] = entry

    def addTotalExcitationLogEntry(self, entry):
        self.total_excitation_time_log[self.current_time] = entry

    def addSigmaCombinationLogEntry(self, entry):
        self.sigma_combination_time_log[self.current_time] = entry

    def moveTimeStep(self):
        self.current_time += self.run_resolution

    def evalulateEntry(self, entry: DensityMatrix):
        validEigen = np.all(
            [val >= -1e-2 for val in entry.eigenvalues().eigenvalues])
        validTrace = round(entry.trace(), 5) == 1

        if not validEigen:
            print(
                "Warning: entry at t={} has negative eigenvalues of {}".format(
                    round(self.current_time, 5),
                    [val for val in entry.eigenvalues().eigenvalues if val <= -1e-15]
                )
            )

        if not validTrace:
            print(
                "Warning: entry at t={} has trace of {}".format(
                    self.current_time, entry.trace()
                )
            )

    def plotRes(self, off_diag=False, legend=True):
        self.res_fig = plotReservoir(self.partial_time_log, off_diag, legend)

    def plotResAndSys(self, off_diag=False, legend=True):
        self.res_sys_fig = plotReservoirAndSystem(self.time_log, off_diag, legend)

    def plotExcite(self):
        self.excitation_fig = plotExcitations(self.excitation_time_log)

    def plotTotalExcite(self, legend=True):
        self.total_excitation_fig = plotTotalExcitations(
            self.total_excitation_time_log, legend)

    def plotSigmaCombinations(self, legend=True):
        self.sigma_combination_fig = plotSigmaCombinations(
            self.sigma_combination_time_log, legend)
