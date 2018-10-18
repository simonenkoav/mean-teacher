# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

from itertools import islice, chain

import numpy as np


def training_batches_transform(data, transform, batch_size=100, random=np.random):
    return eternal_batches_transform(data, transform, batch_size, random)


def evaluation_epoch_generator(data, batch_size=100):
    def generate():
        for idx in range(0, len(data), batch_size):
            yield data[idx:(idx + batch_size)]
    return generate


def evaluation_epoch_generator_transform(data, transform, batch_size=100):
    def generate():
        for idx in range(0, len(data), batch_size):
            arr_len = batch_size
            if idx + batch_size > len(data):
                arr_len = len(data) - idx
            transformed_batch = np.zeros(arr_len, dtype=[
                ('x', np.float32, (32, 32, 3)),
                ('y_24', np.int32, ()),  # We will be using -1 for unlabeled
                ('y_3', np.int32, ())
            ])

            x_data, y_data_24, y_data_3 = [], [], []
            for d in data[idx:(idx + batch_size)]:
                iname, label_24, label_3 = d
                image = transform(iname)
                x_data.append(image)
                y_data_24.append(label_24)
                y_data_3.append(label_3)

            transformed_batch['x'] = x_data
            transformed_batch['y_24'] = y_data_24
            transformed_batch['y_3'] = y_data_3
            yield transformed_batch
    return generate


def training_batches(data, batch_size=100, n_labeled_per_batch='vary', random=np.random):
    if n_labeled_per_batch == 'vary':
        return eternal_batches(data, batch_size, random)
    elif n_labeled_per_batch == batch_size:
        labeled_data, _ = split_labeled(data)
        return eternal_batches(labeled_data, batch_size, random)
    else:
        assert 0 < n_labeled_per_batch < batch_size
        n_unlabeled_per_batch = batch_size - n_labeled_per_batch
        labeled_data, _ = split_labeled(data)
        return combine_batches(
            eternal_batches(labeled_data, n_labeled_per_batch, random),
            unlabel_batches(eternal_batches(data, n_unlabeled_per_batch, random))
        )


def split_labeled(data):
    is_labeled = (data['y'] != -1)
    return data[is_labeled], data[~is_labeled]


def combine_batches(*batch_generators):
    return (np.concatenate(batches) for batches in zip(*batch_generators))


def eternal_batches(data, batch_size=100, random=np.random):
    assert batch_size > 0 and len(data) > 0
    for batch_idxs in eternal_random_index_batches(len(data), batch_size, random):
        yield data[batch_idxs]


def eternal_batches_transform(data, transform, batch_size=100, random=np.random):
    assert batch_size > 0 and len(data) > 0
    for batch_idxs in eternal_random_index_batches(len(data), batch_size, random):
        transformed_batch = np.zeros(len(batch_idxs), dtype=[
            ('x', np.float32, (32, 32, 3)),
            ('y_24', np.int32, ()),  # We will be using -1 for unlabeled
            ('y_3', np.int32, ())
        ])

        x_data, y_data_24, y_data_3 = [], [], []
        for d in data[batch_idxs]:
            iname, label_24, label_3 = d
            image = transform(iname)
            x_data.append(image)
            y_data_24.append(label_24)
            y_data_3.append(label_3)

        transformed_batch['x'] = x_data
        transformed_batch['y_24'] = y_data_24
        transformed_batch['y_3'] = y_data_3

        yield transformed_batch


def unlabel_batches(batch_generator):
    for batch in batch_generator:
        batch["y"] = -1
        yield batch


def eternal_random_index_batches(max_index, batch_size, random=np.random):
    def random_ranges():
        while True:
            indices = np.arange(max_index)
            random.shuffle(indices)
            yield indices

    def batch_slices(iterable):
        while True:
            yield np.array(list(islice(iterable, batch_size)))

    eternal_random_indices = chain.from_iterable(random_ranges())
    return batch_slices(eternal_random_indices)
