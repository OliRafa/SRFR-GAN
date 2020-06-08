import logging

import numpy as np
import tensorflow as tf
from scipy import interpolate
from scipy.optimize import brentq
from sklearn import metrics

from training.metrics import normalize
from validation.lfw_helper import evaluate

LOGGER = logging.getLogger(__name__)


def _predict(images_batch, images_aug_batch, model):
    _, embeddings = model(images_batch, False, 'syn')
    _, embeddings_augmented = model(images_aug_batch, False, 'syn')
    embeddings = embeddings + embeddings_augmented

    if np.all(embeddings.numpy() == 0):
        # If the array is full of 0's, the following calcs that will use it
        # will return NaNs or 0, and at some point the validation algorithm will
        # crash at some division because of the NaNs or the 0's.
        # To not fall on this path, we will insert a value very close to 0
        # instead.
        embeddings = tf.convert_to_tensor(
            np.full_like(embeddings.numpy(), 10e-8),
            dtype=tf.float32,
        )

    return normalize(embeddings, axis=1, name='lfw_normalize_embeddings')


def _predict_on_batch(strategy, model, dataset, dataset_augmented):
    embeddings = np.array([])
    for images_batch, images_aug_batch in zip(dataset, dataset_augmented):
        embedding_per_replica = strategy.experimental_run_v2(
            _predict,
            args=(images_batch, images_aug_batch, model)
        )
        # `embedding_per_replica` is a tuple of EagerTensors, and each tensor
        # has a shape of [batch_size, 512], so we need to get each EagerTensor
        # that is in the tuple, and get each single embedding array that is in
        # the outputted array, appending each of them to the embeddings list.
        for tensor in embedding_per_replica.values:

            # Some tensors have NaNs among the values, so we check them and add
            # 0s in its places
            np_tensor = tensor.numpy()
            if np.isnan(np.sum(np_tensor)):
                tensor = tf.convert_to_tensor(
                    np.nan_to_num(np_tensor),
                    dtype=tf.float32,
                )

            if embeddings.size == 0:
                embeddings = tensor.numpy()
            else:
                try:
                    embeddings = np.concatenate((embeddings, tensor.numpy()),
                                                axis=0)

                # Sometimes the outputted embedding array isn't in the shape of
                # (batch_size, embedding_size), so we need to expand_dims
                # to transform this array from (embedding_size, ) to
                # (1, embedding_size) before concatenatting with `embeddings`
                except ValueError:
                    new_embeddings = np.expand_dims(tensor.numpy(), axis=0)
                    embeddings = np.concatenate((embeddings, new_embeddings),
                                                axis=0)

    return embeddings


def _get_embeddings(strategy, model, left_pairs, left_aug_pairs, right_pairs,
                    right_aug_pairs, is_same_list):
    left_pairs = _predict_on_batch(strategy, model, left_pairs, left_aug_pairs)
    right_pairs = _predict_on_batch(strategy, model, right_pairs,
                                    right_aug_pairs)

    embeddings = []
    for left, right in zip(left_pairs, right_pairs):
        embeddings.append(left)
        embeddings.append(right)

    return np.array(embeddings), is_same_list


def validate_model_on_lfw(strategy, model, left_pairs, left_aug_pairs,
                          right_pairs, right_aug_pairs, is_same_list) -> float:
    """Validates the given model on the Labeled Faces in the Wild dataset.

    ### Parameters
        model: The model to be tested.
        dataset: The Labeled Faces in the Wild dataset, loaded from load_lfw\
 function.
        pairs: List of LFW pairs, loaded from load_lfw_pairs function.

    ### Returns
        (accuracy_mean, accuracy_std, validation_rate, validation_std, far,\
 auc, eer) - Accuracy Mean, Accuracy Standard Deviation, Validation Rate,\
 Validation Standard Deviation, FAR, Area Under Curve (AUC) and Equal Error\
 Rate (EER).
    """
    embeddings, is_same_list = _get_embeddings(strategy, model, left_pairs,
                                               left_aug_pairs, right_pairs,
                                               right_aug_pairs, is_same_list)

    tpr, fpr, accuracy, val, val_std, far = evaluate(embeddings, is_same_list)
    auc = metrics.auc(fpr, tpr)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    return np.mean(accuracy), np.std(accuracy), val, val_std, far, auc, eer
