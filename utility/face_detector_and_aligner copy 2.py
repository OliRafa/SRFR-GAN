import os
import sys
import logging
import glob
from functools import partial
from concurrent import futures
import multiprocessing as mp
import itertools as it

import cv2
import numpy as np
from mtcnn import MTCNN
from skimage import transform
from more_itertools import ichunked
from functional_error_handling import Result, bind

NUM = 240_585 + 10_730 + 194_890 + 88_970 + 90_040 + 76_000 + 6_400

sys.stdout = open('TEST_fda_default_more_cores.txt', 'a')

logging.basicConfig(
    filename='TEST_fda_default_more_cores_logs.txt',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

images = []

@bind
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
        aligned_image = cv2.warpAffine(
            image,
            transformation_matrix,
            crop_shape,
            borderValue=0.0
        )
        return Result(
            'Success',
            (aligned_image, path)
        )
    except Exception as exception:
        return Result(
            'Failure',
            'An error occurred, probably transformation_matrix is None -\
                Error: {}'.format(str(exception))
        )

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

def _mtcnn_detect_faces(image, path):
    import tensorflow as tf
    gpus = tf.config.experimental.get_visible_devices()
    tf.config.experimental.set_memory_growth(gpus[-2], True)
    tf.config.experimental.set_memory_growth(gpus[-1], True)
    return MTCNN().detect_faces(image), path

@bind
def _detect_faces(image, path, queue):
    """Detects faces in a given image and returns it's bounding boxes and\
        facial landmarks.

    ## Parameters
        image: image to be searched on.

    ## Returns
        (bounding_box, facial_landmarks) for the face.
    """
    print('_detect_faces - PID: {}'.format(os.getpid()))
    queue.put(_mtcnn_detect_faces(image, path))
    detected_faces = queue.get()

    detected_faces, path = detected_faces

    if not detected_faces:
        return Result(
            'Failure',
            ' File: {} had zero faces detected.'.format(path)
        )

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

    return Result(
        'Success',
        (image, path, facial_landmarks)
    )

@bind
def _save_image(image, folder, file_name, destination_path):
    """
    """
    print('_save_image - PID: {}'.format(os.getpid()))

    destination_path = os.path.join(destination_path, folder)
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    destination_path = os.path.join(destination_path, file_name + '.jpg')

    if cv2.imwrite(
            destination_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ):
        return Result(
            'Success',
            ' Created file - Path: {}'.format(destination_path)
        )
    return Result(
        'Failure',
        ' File {} not saved'.format(destination_path)
    )

@bind
def _split_file_path(image, file_path):
    """
    """
    print('_process_path - PID: {}'.format(os.getpid()))
    folder = os.path.dirname(file_path).rsplit('/', 1)[1]
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    return Result(
        'Success',
        (image, folder, file_name)
    )

def _read_image(file_path):
    """
    """
    print('_read_image - PID: {}'.format(os.getpid()))
    #class_id, sample = _split_file_path(file_path)
    #image = cv2.imread(file_path)
    #if not image:
    #    print('ué')
    #    logger.error(
    #        " Image couldn't be loaded - path: {}".format(file_path)
    #    )
    #    return None
    #print('foi')
    #return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), '{}/{}'.format(class_id, sample)

    try:
        image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        return Result('Success', (image, file_path))
    except OSError as exception:
        return Result(
            'Failure',
            " Image couldn't be loaded - path: {}, Exception: {}".format(
                file_path,
                str(exception)
            )
        )

def _log_results(Result):
    """
    """
    if Result.get_result() == 'Success':
        logger.info(
            Result.get_payload()
        )
        return True
    else:
        logger.error(
            Result.get_payload()
        )
    return False

def _preprocess_pipeline(file_path, destination_folder):
    """
    """
    print('detect_and_align_faces - PID: {}'.format(os.getpid()))

    raw_image = _read_image(file_path)
    detected_image = _detect_faces(raw_image, queue=queue)
    cropped_face = _align_face(detected_image)
    file_path_splitted = _split_file_path(cropped_face)
    result = _save_image(file_path_splitted, destination_folder)
    return _log_results(result)

def detect_and_align_faces(dataset_folder, destination_folder):
    """
    """
    import timing
    mp.set_start_method('spawn')

    #manager = mp.Manager()
    #queue = manager.Queue()

    _preprocess = partial(
        _preprocess_pipeline,
        destination_folder=destination_folder,
    #    queue=queue
    )

    dataset_folder = glob.iglob(dataset_folder)
    dataset_folder = it.islice(dataset_folder, NUM, None)
    batches = ichunked(dataset_folder, 1000)

    for batch in batches:
        #with futures.ProcessPoolExecutor() as executor:
        #    executor.map(_preprocess, batch)
        tuple(map(memory_test, batch))
        print(images)
        input()

def memory_test(file_path):
    images.append(_read_image(file_path))

if __name__ == '__main__':
    DATASET_FOLDER = '/mnt/hdd_raid/datasets/VGGFace2/train/*/*'
    DESTINATION_FOLDER = '/mnt/hdd_raid/datasets/VGGFace2_Aligned/train/'
    #DESTINATION_FOLDER = '/mnt/hdd_raid/datasets/TESTE/t2/'

    detect_and_align_faces(DATASET_FOLDER, DESTINATION_FOLDER)
