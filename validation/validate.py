import os
import sys

import numpy as np
import tensorflow as tf
from scipy import interpolate
from scipy.optimize import brentq
from sklearn import metrics
from input_data import (
    augment_dataset_without_concatenation,
    load_dataset,
    normalize_images,
)
from input_data import split_path
from training.losses import normalize
from validation.lfw_helper import evaluate

# _load_pairs dar load no class_id e no sample_name separados
# Verificar funções no evaluate
def _load_pairs():
    pairs = []
    with open('pairs_label.txt', 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            _, id_01 = split_path(pair[0])
            _, id_02 = split_path(pair[1])
            pairs.append([id_01, id_02, pair[2]])
    return np.array(pairs)

@tf.function
def _get_single_embedding(model, dataset):
    image, augmented_image = dataset.map(lambda a, b, c, d: (a, b))
    embeddings, _ = model(image)
    embeddings_augmented, _ = model(augmented_image)
    embeddings = embeddings + embeddings_augmented
    return normalize(embeddings, name='lfw_normalize_embeddings')

@tf.function
def _get_embeddings(model, dataset):
    embeddings = []
    is_same_list = []
    pairs = _load_pairs()
    for id_01, id_02, is_same in pairs:
        id_01_dataset = dataset.filter(lambda x: tf.equal(x[3], id_01))
        id_02_dataset = dataset.filter(lambda x: tf.equal(x[3], id_02))

        embeddings.append(_get_single_embedding(model, id_01_dataset))
        embeddings.append(_get_single_embedding(model, id_02_dataset))

        is_same_list.append(is_same)
    return embeddings, is_same_list

@tf.function
def validate_model_on_lfw(model, dataset) -> float:
    """Validates the given model on the Labeled Faces in the Wild dataset.

    ### Parameters
        model: The model to be tested.
        dataset: The Labeled Faces in the Wild dataset, loaded from load_lfw\
 function.

    ### Returns
        (accuracy_mean, accuracy_std, validation_rate, validation_std, far,\
 auc, eer) - Accuracy Mean, Accuracy Standard Deviation, Validation Rate,\
 Validation Standard Deviation, FAR, Area Under Curve (AUC) and Equal Error\
 Rate (EER).
    """
    
    embeddings, is_same_list = _get_embeddings(model, dataset)

    tpr, fpr, accuracy, val, val_std, far = evaluate(embeddings, is_same_list)
    auc = metrics.auc(fpr, tpr)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    return np.mean(accuracy), np.std(accuracy), val, val_std, far, auc, eer

if __name__ == '__main__':
    pairs = _load_pairs()
    print(pairs.shape)