import sys
import numpy as np
from numpy import mean, cov, dot, linalg, transpose
from McNeuron import Neuron
from McNeuron import Node
from McNeuron import visualize
from __builtin__ import str
from copy import deepcopy

class collection(object):

    def __init__(dataset):
        self.database = database
        self.n = len(self.database)
        
    def register_from_file(self, database):
        self.database = []
        for i in range(self.n):
            n = Neuron(file_format = 'swc', input_file = database[i])
            self.database.append(n)

    def set_hist_range(self, hist_range):
        """
        set the range of histogram for each feature.

        hist_range : dict
        ----------
            dictionary of all feature and thier range of histogram.
        """
        self.feature_name = hist_range.keys()
        self.hist_range = hist_range

    def mean(self):
        self.mean = {}
        for name in self.feature_name:

            self.mean[name] = 1


    #def variance():
