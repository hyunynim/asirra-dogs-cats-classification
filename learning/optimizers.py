from abc import abstractmethod
import tensorflow as tf
import numpy as np
import time


class Optimizer(object):
    """
    Base class for optimization functions.
    """

    def __init__(self, sess, model, train_set, evaluator, **kwargs):
        """
        Optimizer initializer.
        :param sess: tf.Session.
        :param model: Model to be learned.
        :param train_set: DataSet.
        """
        self.sess = sess
        self.model = model
        self.train_set = train_set
        self.evaluator = evaluator

        self.batch_size = kwargs.pop('batch_size', 256)
        self.num_epochs = kwargs.pop('num_epochs', 320)
        self.init_learning_rate = kwargs.pop('learning_rate', 0.01)

        self.learning_rate = tf.placeholder(tf.float32)

    def _reset(self):
        """Reset some variables."""
        self.curr_epoch = 1
        self.best_performance = self.evaluator.worst_performance()
        self.saver = tf.train.Saver()

        self.curr_learning_rate = self.init_learning_rate    # current learning rate
        self.optimize = self._optimize_op()

        # Initialize all weights
        self.sess.run(tf.global_variables_initializer())

    @abstractmethod
    def _optimize_op(self):
        """
        tf.train.Optimizer.minimize Op for a gradient update.
        This should be implemented, and should not be called manually.
        """
        pass

    def _step(self):
        """
        Make a single gradient update and return its results.
        This should not be called manually.
        """
        step_results = dict()
        # Sample a single batch
        X, y_true = self.train_set.next_batch(self.batch_size, shuffle=True, augment=True)

        # Compute the loss and make update
        _, loss, y_pred = \
            self.sess.run([self.optimize, self.model.loss, self.model.predict],
                          feed_dict={self.model.X: X, self.model.y: y_true,
                                     self.model.is_train: True,
                                     self.learning_rate: self.curr_learning_rate})
        step_results['loss'] = loss
        step_results['y_true'] = y_true
        step_results['y_pred'] = y_pred

        return step_results

    @abstractmethod
    def _update_learning_rate(self):
        """
        Update current learning rate, by its own schedule.
        This should be implemented, and should not be called manually.
        """

    def train(self, details=False, eval=True, verbose=True):
        """
        Run optimizer to train the model.
        :param eval: Boolean, whether to perform evaluation on every epoch.
        """
        train_results = dict()    # dictionary to contain training results and details
        train_size = self.train_set.num_examples
        num_steps_per_epoch = train_size // self.batch_size
        num_steps = self.num_epochs * num_steps_per_epoch

        if verbose:
            print('Running training loop...')
            print('Number of training iterations: {}'.format(num_steps))

        step_losses, step_performances, eval_performances = [], [], []
        start_time = time.time()
        for i in range(num_steps):
            step_results = self._step()
            step_loss = step_results['loss']
            step_losses.append(step_loss)

            if (i+1) % num_steps_per_epoch == 0:
                step_performance = self.evaluator.performance(step_results['y_true'], step_results['y_pred'])
                step_performances.append(step_performance)
                if eval:
                    # Evaluate current model
                    eval_performance = self.evaluator.eval()['performance']
                    eval_performances.append(eval_performance)

                    if verbose:
                        print('[epoch {}]\tloss: {} |Train performance: {} |Eval performance: {} |learning_rate: {}'\
                              .format(self.curr_epoch, step_loss,
                                      step_performance, eval_performance, self.curr_learning_rate))

                    # Keep track of the current best model for evaluation set
                    if self.evaluator.is_better(eval_performance, self.best_performance):
                        self.best_performance = eval_performance
                        self.saver.save(self.sess, '/tmp/model.ckpt')    # save current weights

                else:
                    if verbose:
                        print('[epoch {}]\tloss: {} |Train performance: {} |learning_rate: {}'\
                              .format(self.curr_epoch, step_loss,
                                      step_performance, self.curr_learning_rate))

                    # Keep track of the current best model for training set
                    if self.evaluator.is_better(step_performance, self.best_performance):
                        self.best_performance = step_performance
                        self.saver.save(self.sess, '/tmp/model.ckpt')    # save current weights

                self._update_learning_rate()
                self.curr_epoch += 1

        if verbose:
            print('Total training time(sec): {}'.format(time.time() - start_time))
            print('Best {} performance: {}'.format('evaluation' if eval else 'training',
                                                   self.best_performance))

        if details:
            # Store training results in a dictionary
            train_results['step_losses'] = step_losses    # (num_iterations)
            train_results['step_performances'] = step_performances    # (num_epochs)
            if eval:
                train_results['eval_performances'] = eval_performances    # (num_epochs)

            return details





















