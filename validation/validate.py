import logging

import numpy as np
import tensorflow as tf
from scipy import interpolate
from scipy.optimize import brentq
from sklearn import metrics
from tqdm import tqdm

from training.metrics import normalize
from validation.lfw_helper import evaluate

LOGGER = logging.getLogger(__name__)


def _get_single_embedding(model, dataset):
    image, augmented_image, _ = list(*dataset)
    embeddings, _ = model(tf.expand_dims(image, 0))
    embeddings_augmented, _ = model(tf.expand_dims(augmented_image, 0))
    embeddings = embeddings + embeddings_augmented
    return normalize(embeddings, name='lfw_normalize_embeddings')


def _get_embeddings(model, dataset, pairs):
    embeddings = []
    is_same_list = []
    for id_01, id_02, is_same in tqdm(pairs):
        id_01_dataset = dataset.filter(lambda x, y, z: tf.equal(z, id_01))
        id_02_dataset = dataset.filter(lambda x, y, z: tf.equal(z, id_02))

        embeddings.append(_get_single_embedding(model, id_01_dataset))
        embeddings.append(_get_single_embedding(model, id_02_dataset))

        is_same_list.append(is_same)
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
    embeddings, is_same_list = _get_embeddings(model, dataset, pairs)

    tpr, fpr, accuracy, val, val_std, far = evaluate(embeddings, is_same_list)
    auc = metrics.auc(fpr, tpr)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    return np.mean(accuracy), np.std(accuracy), val, val_std, far, auc, eer
