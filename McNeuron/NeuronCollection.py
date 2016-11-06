import sys
import numpy as np
from numpy import mean, cov, dot, linalg, transpose
from __builtin__ import str
from copy import deepcopy

class collection(object):

    def __init__(dataset):
        self.dataset = dataset

    def set_hist_range(hist_range):
        """
        set the range of histogram for each feature.

        hist_range : dict
        ----------
            dictionary of all feature and thier range of histogram.
        """
        self.feature_name = hist_range.keys()
        self.hist_range = hist_range

    def mean():
        self.mean = {}
        for name in self.feature_name:

            self.mean[name] = 1
    #def variance():
