# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""CIFAR-10 final evaluation"""

import logging
import sys

from experiments.run_context import RunContext
import tensorflow as tf

from datasets.cifar10 import Eye24ZCA
from mean_teacher.model import Model
from mean_teacher import minibatching
from datasets.image_utils import train_pipeline, eval_pipeline


LOG = logging.getLogger('main')


def parameters():
    test_phase = True
    for model_type in ['mean_teacher']:
        n_runs = 1
        for data_seed in range(2000, 2000 + n_runs):
            yield {'test_phase': test_phase,
                'model_type': model_type,
                'data_seed': data_seed
            }


def model_hyperparameters(model_type, n_labeled, n_all):
    assert model_type in ['mean_teacher', 'pi']
    if n_labeled == 'all':
        return {
            'n_labeled_per_batch': 100,
            'max_consistency_cost': 100.0,
            'apply_consistency_to_labeled': True,
            'ema_consistency': model_type == 'mean_teacher'
        }
    elif isinstance(n_labeled, int):
        return {
            'n_labeled_per_batch': 'vary',
            'max_consistency_cost': 100.0 * n_labeled / n_all,
            'apply_consistency_to_labeled': True,
            'ema_consistency': model_type == 'mean_teacher'
        }
    else:
        msg = "Unexpected combination: {model_type}, {n_labeled}"
        assert False, msg.format(locals())


def run(test_phase, data_seed, model_type):
    minibatch_size = 100
    # fixed and pre-calculated (from file)
    n_labeled = 27360
    n_all = 44458
    hyperparams = model_hyperparameters(model_type, n_labeled, n_all)

    tf.reset_default_graph()
    model = Model(RunContext(__file__, data_seed))

    # train_filename = '/root/storage/hdd/eyes_color/descriptions_files/labeled_unlabeled_vgg_35k.txt'
    # test_filename = '/root/storage/hdd/eyes_color/descriptions_files/test_base_path.txt'
    eye_dataset = Eye24ZCA(n_labeled=n_labeled,
                           data_seed=data_seed,
                           test_phase=test_phase)

    model['flip_horizontally'] = True
    model['ema_consistency'] = hyperparams['ema_consistency']
    model['max_consistency_cost'] = hyperparams['max_consistency_cost']
    model['apply_consistency_to_labeled'] = hyperparams['apply_consistency_to_labeled']
    model['adam_beta_2_during_rampup'] = 0.999
    model['ema_decay_during_rampup'] = 0.999
    model['normalize_input'] = False  # Keep ZCA information # TODO not sure
    model['rampdown_length'] = 25000
    model['training_length'] = 200000

    training_batches = minibatching.training_batches(eye_dataset.training, minibatch_size,
                                                     hyperparams['n_labeled_per_batch'])
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(eye_dataset.evaluation, minibatch_size)

    tensorboard_dir = model.save_tensorboard_graph()
    LOG.info("Saved tensorboard graph to %r", tensorboard_dir)

    model.train(training_batches, evaluation_batches_fn)


if __name__ == "__main__":
    for run_params in parameters():
        run(**run_params)
