import numpy as np
import tensorflow as tf
from scipy import interpolate
from scipy.optimize import brentq
from sklearn import metrics

from training.losses import normalize
from validation.lfw_helper import evaluate

# Talvez no InputData.split_path dentro do _load_pairs tenha que retirar o @tf.function
@tf.function
def _get_single_embedding(model, dataset):
    image, augmented_image = dataset.map(lambda a, b, c, d: (a, b))
    embeddings, _ = model(image)
    embeddings_augmented, _ = model(augmented_image)
    embeddings = embeddings + embeddings_augmented
    return normalize(embeddings, name='lfw_normalize_embeddings')

@tf.function
def _get_embeddings(model, dataset, pairs):
    embeddings = []
    is_same_list = []
    for id_01, id_02, is_same in pairs:
        id_01_dataset = dataset.filter(lambda x: tf.equal(x[2], id_01))
        id_02_dataset = dataset.filter(lambda x: tf.equal(x[2], id_02))

        embeddings.append(_get_single_embedding(model, id_01_dataset))
        embeddings.append(_get_single_embedding(model, id_02_dataset))

        is_same_list.append(is_same)
    return embeddings, is_same_list

@tf.function
def validate_model_on_lfw(model, dataset, pairs) -> float:
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
