import os
import sys

import tensorflow as tf
import numpy as np
from input_data import augment_dataset, load_dataset, normalize_images
from lfw_helper import evaluate
from scipy import interpolate
from scipy.optimize import brentq
from sklearn import metrics
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training'))
from losses import normalize

# Get LFW
# Get embeddings for the faces (with the flipped ones)
# Get the mean for each pair of embeddings (face, flipped_face)
# Calculate ROC
# Get AUC -> scikit-learn
# Print acc and draw graph

# _load_pairs dar load no class_id e no sample_name separados
# Ajustar o algoritmo para trabalhar com batchs em vez de single examples
# Verificar funções no evaluate
def _load_pairs():
    pairs = []
    with open('file', r) as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pais)

def validate_model(model) -> float:
    dataset, _, _ = load_dataset('LFW', remove_overlaps=False, sample_ids=True)
    dataset = dataset.map(augment_dataset)
    dataset = dataset.map(
        lambda image, class_id, sample: (
            normalize_images(image),
            class_id,
            sample
        )
    )
    pairs = _load_pairs()
    for id_01, id_02 in pairs:
        id_01 = dataset.filter(
            lambda image, class_id, sample: image if sample == id_01
        )
        id_02 = dataset.filter(
            lambda image, class_id, sample: image if sample == id_02
        )
        id_01_augmented = dataset.filter(
            lambda image, class_id, sample: image if sample == id_01+"-augmented"
        )
        id_02_augmented = dataset.filter(
            lambda image, class_id, sample: image if sample == id_02+"-augmented"
        )
        embeddings_01, _ = model(id_01)
        embeddings_01_augmented, _ = model(id_01_augmented)
        embeddings_02, _ = model(id_02)
        embeddings_02_augmented, _ = model(id_02_augmented)

        embeddings_01 = embeddings_01 + embeddings_01_augmented
        embeddings_02 = embeddings_02 + embeddings_02_augmented

        embeddings_01 = normalize(embeddings_01, name='lfw_normalize_embeddings')
        embeddings_02 = normalize(embeddings_02, name='lfw_normalize_embeddings')




    tpr, fpr, accuracy, val, val_std, far = evaluate()
    auc = metrics.auc(fpr, tpr)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    return np.mean(accuracy), np.std(accuracy), val, val_std, far, auc, eer
