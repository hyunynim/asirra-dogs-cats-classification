from abc import abstractmethod, abstractproperty
import tensorflow as tf
import numpy as np
import time


class Evaluator(object):
    """
    Base class for evaluation functions.
    """

    @abstractproperty
    def worst_score(self):
        """The worst performance score."""
        pass

    @abstractmethod
    def score(self, y_true, y_pred):
        """
        Performance metric for a given prediction.
        This should be implemented.
        :param y_true: np.ndarray, shape: (N, num_classes).
        :param y_pred: np.ndarray, shape: (N, num_classes).
        """
        pass

    @abstractmethod
    def is_better(self, curr, best):
        """
        Function to return whether current performance score is better than current best.
        This should be implemented.
        :param curr: Float, currently given performance.
        :param best: Float, current best performance.
        """
        pass
