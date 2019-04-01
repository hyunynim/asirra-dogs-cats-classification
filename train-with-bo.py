import os
import numpy as np
import tensorflow as tf
from datasets import asirra as dataset
from models.nn import AlexNet as ConvNet
from learning.optimizers import MomentumOptimizer as Optimizer
from learning.evaluators import AccuracyEvaluator as Evaluator
from bayes_opt import BayesianOptimization


""" 1. Load and split datasets """
root_dir = os.path.join('/', 'mnt', 'sdb2', 'Datasets', 'asirra')    # FIXME
trainval_dir = os.path.join(root_dir, 'train')

# Load trainval set and split into train/val sets
X_trainval, y_trainval = dataset.read_asirra_subset(trainval_dir, one_hot=True)
trainval_size = X_trainval.shape[0]
val_size = int(trainval_size * 0.2)    # FIXME
val_set = dataset.DataSet(X_trainval[:val_size], y_trainval[:val_size])
train_set = dataset.DataSet(X_trainval[val_size:], y_trainval[val_size:])

# Sanity check
print('Training set stats:')
print(train_set.images.shape)
print(train_set.images.min(), train_set.images.max())
print((train_set.labels[:, 1] == 0).sum(), (train_set.labels[:, 1] == 1).sum())
print('Validation set stats:')
print(val_set.images.shape)
print(val_set.images.min(), val_set.images.max())
print((val_set.labels[:, 1] == 0).sum(), (val_set.labels[:, 1] == 1).sum())


""" 2. Set training hyperparameters (not to be optimized) """
hp_d = dict()
image_mean = train_set.images.mean(axis=(0, 1, 2))    # mean image
np.save('/tmp/asirra_mean.npy', image_mean)    # save mean image
hp_d['image_mean'] = image_mean

# FIXME: Training hyperparameters
hp_d['batch_size'] = 256
hp_d['num_epochs'] = 200

hp_d['augment_train'] = True
hp_d['augment_pred'] = True

hp_d['momentum'] = 0.9
hp_d['learning_rate_patience'] = 30
hp_d['learning_rate_decay'] = 0.1
hp_d['eps'] = 1e-8

# FIXME: Regularization hyperparameters
hp_d['dropout_prob'] = 0.5

# FIXME: Evaluation hyperparameters
hp_d['score_threshold'] = 1e-4


""" 3. Define function for training and validating deep neural networks(DNNs) once """
def train_and_validate(init_learning_rate_log, weight_decay_log):
    tf.reset_default_graph()
    graph = tf.get_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    hp_d['init_learning_rate'] = 10**init_learning_rate_log
    hp_d['weight_decay'] = 10**weight_decay_log

    model = ConvNet([227, 227, 3], 2, **hp_d)
    evaluator = Evaluator()
    optimizer = Optimizer(model, train_set, evaluator, val_set=val_set, **hp_d)

    sess = tf.Session(graph=graph, config=config)
    train_results = optimizer.train(sess, details=True, verbose=True, **hp_d)

    # Return the maximum validation score as target
    best_val_score = np.max(train_results['eval_scores'])

    return best_val_score


""" 4. Start Bayesian Optimization for hyperparameter optimization """
bayes_optimizer = BayesianOptimization(
    f=train_and_validate,
    pbounds={
        'init_learning_rate_log': (-5, -1),    # FIXME
        'weight_decay_log': (-5, -1)           # FIXME
    },
    random_state=0,
    verbose=2
)

bayes_optimizer.maximize(init_points=3, n_iter=27, acq='ei', xi=0.01)    # FIXME

# Print final results
for i, res in enumerate(bayes_optimizer.res):
    print('Iteration {}: \n\t{}'.format(i, res))
print('Final result: ', bayes_optimizer.max)
