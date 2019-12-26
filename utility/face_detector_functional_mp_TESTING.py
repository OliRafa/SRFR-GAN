import os
import sys
import logging
from glob import glob
from concurrent import futures
import multiprocessing as mp
import itertools as it

import cv2
import numpy as np
from mtcnn import MTCNN
from skimage import transform

sys.stdout = open('face_detector_functional_mp_TESTING.txt', 'a')

logging.basicConfig(
    filename='face_detector_functional_mp_TESTING_logs.txt',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def _align_face(
        image,
        path,
        facial_landmarks,
        crop_shape=(112, 112)
    ):
    """Align faces using the facial landmarks or the bounding box and crops
    them.

    ## Parameters
        image: face image to be aligned.
        facial_landmarks: facial landmarks for the face in the image.
        bounding_box: optional bounding box.
        crop_shape: optional shape for the crop.

    ## Returns
        Face image aligned and cropped.
    """
    print('_align_faces - PID: {}'.format(os.getpid()))

    source_landmarks = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)
    if crop_shape == (112, 112):
        source_landmarks[:, 0] += 8.0

    facial_landmarks = np.asfarray(facial_landmarks)

    transformation = transform.SimilarityTransform()
    transformation.estimate(facial_landmarks, source_landmarks)
    transformation_matrix = transformation.params[0:2, :]

    try:
        return cv2.warpAffine(
            image,
            transformation_matrix,
            crop_shape,
            borderValue=0.0
        ), path
    except Exception as e:
        logger.error(
            'An error occurred, probably transformation_matrix is None -\
                Error: {}'.format(str(e))
            )

    return None

def _calculate_distance_from_center(image_shape, face):
    """
    """
    image_center = np.asarray([image_shape[0] / 2, image_shape[1] / 2])

    bounding_box_mean = np.asarray([
        np.mean([face['box'][0], face['box'][1]]),
        np.mean([face['box'][2], face['box'][3]])
    ])
    return np.linalg.norm(bounding_box_mean - image_center), face

def _extract_center_face(image, detected_faces):
    """Extracts the face that's the closest to the center of a image.

    ## Parameters
        image: image with the faces.
        detected_faces: list of resulting bounding boxes and keypoints from
        MTCNN().detect_faces().

    ## Returns
        (bounding_box, facial_landmarks) for the closest face to the center of
        the image.
    """
    print('_extract_center_face - PID: {}'.format(os.getpid()))

    center_face = min(
        (
            _calculate_distance_from_center(image.shape, face)
            for face in detected_faces
        ),
        key=lambda x: x[0]
    )
    bounding_box = center_face[1]['box']
    facial_landmarks = center_face[1]['keypoints']

    return  bounding_box, facial_landmarks

def _detect_faces(image, path):
    """Detects faces in a given image and returns it's bounding boxes and\
        facial landmarks.

    ## Parameters
        image: image to be searched on.

    ## Returns
        (bounding_box, facial_landmarks) for the face.
    """
    print('_detect_faces - PID: {}'.format(os.getpid()))
    detected_faces = MTCNN().detect_faces(image)

    if not detected_faces:
        logger.warning(' File: {} had zero faces detected.'.format(path))
        return None

    if len(detected_faces) > 1:
        _, keypoints = _extract_center_face(image, detected_faces)
    else:
        #bounding_box = detected_faces[0]['box']
        keypoints = detected_faces[0]['keypoints']

    facial_landmarks = (
        keypoints['left_eye'],
        keypoints['right_eye'],
        keypoints['nose'],
        keypoints['mouth_left'],
        keypoints['mouth_right']
    )

    return image, path, facial_landmarks

def _save_image(image, destination_path, destination_folder):
    """
    """
    print('_save_image - PID: {}'.format(os.getpid()))

    folder, image_path = destination_path.split('/')
    destination_folder = '{}{}/'.format(destination_folder, folder)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    destination_path = '{}{}.jpg'.format(destination_folder, image_path)

    if cv2.imwrite(
        destination_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ):
        logger.info(' Created file - Path: {}'.format(destination_path))
        return True
    else:
        logger.error(' File {} not saved'.format(destination_path))
        return False

def _split_file_path(file_path):
    """
    """
    print('_process_path - PID: {}'.format(os.getpid()))
    import tensorflow as tf
    gpus = tf.config.experimental.get_visible_devices()
    tf.config.experimental.set_memory_growth(gpus[-2], True)
    tf.config.experimental.set_memory_growth(gpus[-1], True)

    parts = tf.strings.split(file_path, os.path.sep)

    class_id = parts.numpy()[-2].decode('utf-8')
    sample = parts.numpy()[-1].decode('utf-8').split('.')[0]

    return class_id, sample

def _read_image(file_path):
    """
    """
    print('_read_image - PID: {}'.format(os.getpid()))
    class_id, sample = _split_file_path(file_path)

    image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    if not image:
        logger.error(
            " File - path: {} couldn't be loaded".format(file_path)
        )
        return None

    return image, '{}/{}'.format(class_id, sample)

def _preprocess_face(zip_paths):
    """
    """
    file_path, destination_folder = zip_paths
    print('detect_and_align_faces - PID: {}'.format(os.getpid()))
    raw_image = _read_image(file_path)
    detected_image = _detect_faces(*raw_image) if raw_image else None
    cropped_face = _align_face(*detected_image) if detected_image else None
    return _save_image(
        *cropped_face,
        destination_folder
    ) if cropped_face else None

def detect_and_align_faces(dataset_folder, destination_folder):
    """
    """
    import timing
    mp.set_start_method('spawn')

    num = 5_500
    dataset_folder = tuple(glob(dataset_folder)[num:])
    dataset_size = len(dataset_folder)
    print('Dataset len: {}'.format(dataset_size))

    destination_folder = (destination_folder, ) * dataset_size
    dataset_folder = zip(dataset_folder, destination_folder)
    batchs = map(
        lambda x: it.islice(dataset_folder, x, x+5),
        range(0, dataset_size, 5)
    )

    for batch in batchs:
        with futures.ProcessPoolExecutor() as executor:
            executor.map(detect_and_align_faces, batch)

if __name__ == '__main__':
    DATASET_FOLDER = '/mnt/hdd_raid/datasets/VGGFace2/train/*/*'
    #DESTINATION_FOLDER = '/mnt/hdd_raid/datasets/VGGFace2_Aligned/train/'
    DESTINATION_FOLDER = '/mnt/hdd_raid/datasets/TESTE/t2/'

    detect_and_align_faces(DATASET_FOLDER, DESTINATION_FOLDER)
