import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(exp_idx, step_losses, step_scores, eval_scores=None,
                        l_rate_decay_iters_list=None,
                        plot=False, save=True, img_dir='.'):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes[0].plot(np.arange(1, len(step_losses)+1), step_losses, marker='')
    axes[0].set_ylabel('loss')
    axes[0].set_xlabel('Number of iterations')
    axes[1].plot(np.arange(1, len(step_scores)+1), step_scores, color='b', marker='')
    if eval_scores is not None:
        axes[1].plot(np.arange(1, len(eval_scores)+1), eval_scores, color='r', marker='')
    axes[1].set_ylim(0.5, 1.0)  # FIXME
    axes[1].set_ylabel('Accuracy')
    axes[1].set_xlabel('Number of epochs')

    # Show learning rate decay points on learning curve plots
    if l_rate_decay_iters_list is not None:
        for it in l_rate_decay_iters_list:
            axes[0].axvline(x=it, color='k', linestyle='--', linewidth=1)
            axes[1].axvline(x=it / 100, color='k', linestyle='--', linewidth=1)

    if save:
        plot_img_filename = 'learning_curve-result{}.svg'.format(exp_idx)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        fig.savefig(os.path.join(img_dir, plot_img_filename))

        pkl_filename = 'learning_curve-result{}.pkl'.format(exp_idx)
        with open(os.path.join(img_dir, pkl_filename), 'wb') as fo:
            pkl.dump([step_losses, step_scores, eval_scores], fo)

    if plot:
        plt.show()
    else:
        plt.close()
