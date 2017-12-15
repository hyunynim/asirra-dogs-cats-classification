from abc import abstractmethod
import tensorflow as tf
import numpy as np
import time


class Evaluator(object):
    """
    Base class for evaluation functions.
    """

    def __init__(self, sess, model, eval_set, **kwargs):
        """
        Evaluator initializer.
        :param sess: tf.Session.
        :param model: Model to be evaluated.
        :param eval_set: DataSet.
        """
        self.sess = sess
        self.model = model
        self.eval_set = eval_set

        self.batch_size = kwargs.pop('batch_size', 256)

    @abstractmethod
    def worst_performance(self):
        """
        The worst performance score.
        """
        pass

    @abstractmethod
    def performance(self, y_true, y_pred):
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
        Function to return whether current performance is better than current best.
        This should be implemented.
        :param curr: Float, currently given performance.
        :param best: Float, current best performance.
        """
        pass

    def eval(self, details=False, verbose=False):
        """Evaluate the model."""
        eval_results = dict()    # dictionary to contain evaluation results and details
        eval_size = self.eval_set.num_examples
        num_steps = eval_size // self.batch_size

        if verbose:
            print('Running evaluation loop...')

        # Evaluation loop
        eval_y_true, eval_y_pred = [], []
        start_time = time.time()
        for i in range(num_steps+1):
            if i == num_steps:
                _batch_size = eval_size - num_steps*self.batch_size
            else:
                _batch_size = self.batch_size
            X, y_true = self.eval_set.next_batch(_batch_size, shuffle=False, augment=False)

            # Compute predictions
            y_pred = self.sess.run(self.model.predict,
                                   feed_dict={self.model.X: X,
                                              self.model.is_train: False})
            eval_y_true.append(y_true)
            eval_y_pred.append(y_pred)
        if verbose:
            print('Total evaluation time(sec): {}'.format(time.time() - start_time))

        eval_y_true = np.concatenate(eval_y_true, axis=0)    # (N, 2)
        eval_y_pred = np.concatenate(eval_y_pred, axis=0)    # (N, 2)

        # Store evaluation results in a dictionary
        if details:
            eval_results['y_true'] = eval_y_true
            eval_results['y_pred'] = eval_y_pred
        eval_results['performance'] = self.performance(eval_y_true, eval_y_pred)
        return eval_results

