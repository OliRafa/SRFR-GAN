import os
import logging
import glob
from functools import partial
from concurrent import futures
import multiprocessing as mp

import cv2
import numpy as np
from mtcnn import MTCNN
from skimage import transform
from more_itertools import ichunked
from functional_error_handling import bind, ImageContainer, Result

import gc

logging.basicConfig(
    filename='face_detector_and_aligner_logs.txt',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

@bind
def _align_face(image_container, facial_landmarks, crop_shape=(112, 112)):
    """Align faces using the facial landmarks or the bounding box and crops\
 them.

    ## Parameters
        image: face image to be aligned.
        facial_landmarks: facial landmarks for the face in the image.
        bounding_box: optional bounding box.
        crop_shape: optional shape for the crop.

    ## Returns
        Face image aligned and cropped.
    """
    #print('_align_faces - PID: {}'.format(os.getpid()))

    #if crop_shape == (112, 112):
    #    pass
    #else:
    #    pass

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
            image_container.image,
            transformation_matrix,
            crop_shape,
            borderValue=0.0
        )
        return Result(
            'Success',
            ImageContainer(
                image=aligned_image,
                image_path=image_container.image_path
            )
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

def _extract_center_face(image_shape, detected_faces):
    """Extracts the face that's the closest to the center of a image.

    ## Parameters
        image: image with the faces.
        detected_faces: list of resulting bounding boxes and keypoints from\
 MTCNN().detect_faces().

    ## Returns
        (bounding_box, facial_landmarks) for the closest face to the center of\
 the image.
    """
    #print('_extract_center_face - PID: {}'.format(os.getpid()))

    center_face = min(
        (
            _calculate_distance_from_center(image_shape, face)
            for face in detected_faces
        ),
        key=lambda x: x[0]
    )
    bounding_box = center_face[1]['box']
    facial_landmarks = center_face[1]['keypoints']

    return  bounding_box, facial_landmarks

def _mtcnn_detect_faces(image):
    # Imports TF everytime, necessary for multiprocessing the pipeline
    import tensorflow as tf
    gpus = tf.config.experimental.get_visible_devices()
    tf.config.experimental.set_memory_growth(gpus[-2], True)
    tf.config.experimental.set_memory_growth(gpus[-1], True)

    #try:
    #    detected_faces = MTCNN().detect_faces(image)
    #except tf.errors.ResourceExhaustedError:
    #    detected_faces = _mtcnn_detect_faces(image)
    #return detected_faces
    return MTCNN().detect_faces(image)

@bind
def _detect_faces(image_container):
    """Detects faces in a given image and returns it's bounding boxes and\
 facial landmarks.

    ## Parameters
        image: image to be searched on.

    ## Returns
        (bounding_box, facial_landmarks) for the face.
    """
    #print('_detect_faces - PID: {}'.format(os.getpid()))
    detected_faces = _mtcnn_detect_faces(image_container.image)

    if not detected_faces:
        return Result(
            'Failure',
            ' File: {} had zero faces detected.'.format(
                image_container.image_path
            )
        )

    if len(detected_faces) > 1:
        _, keypoints = _extract_center_face(
            image_container.image.shape,
            detected_faces
        )
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
        image_container,
        args=facial_landmarks
    )

def _split_file_path(file_path):
    """
    """
    folder = os.path.dirname(file_path).rsplit('/', 1)[1]
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    return folder, file_name

@bind
def _save_image(image_container, destination_path):
    """
    """
    #print('_save_image - PID: {}'.format(os.getpid()))

    folder, file_name = _split_file_path(image_container.image_path)

    destination_path = os.path.join(destination_path, folder)
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    destination_path = os.path.join(destination_path, file_name + '.jpg')

    if cv2.imwrite(destination_path, cv2.cvtColor(
            image_container.image,
            cv2.COLOR_RGB2BGR
        )):
        return Result(
            'Success',
            ' Created file - Path: {}'.format(destination_path)
        )
    return Result(
        'Failure',
        ' File {} not saved'.format(destination_path)
    )

def _read_image(file_path):
    """
    """
    #print('_read_image - PID: {}'.format(os.getpid()))
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
        return Result(
            'Success',
            ImageContainer(image=image, image_path=file_path)
            )
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
    #print('detect_and_align_faces - PID: {}'.format(os.getpid()))
    gc.unfreeze()

    raw_image = _read_image(file_path)
    detected_image = _detect_faces(raw_image)
    cropped_face = _align_face(detected_image)
    result = _save_image(cropped_face, destination_folder)
    return _log_results(result)

def _compare_folders(dataset_folder, destination_folder):
    dataset_folder = set(glob.glob(dataset_folder + '*/*'))
    destination_folder = set(glob.glob(destination_folder + '*/*'))
    return dataset_folder - destination_folder

def _dataset_generator(dataset_folder, destination_folder):
    dataset = _compare_folders(dataset_folder, destination_folder)

    for file_path in dataset:
        yield file_path

def detect_and_align_faces(dataset_folder, destination_folder):
    """
    """
    import timing
    mp.set_start_method('spawn')

    gc.collect()

    dataset_folder = _dataset_generator(dataset_folder, destination_folder)
    #dataset_folder = it.islice(dataset_folder)
    batches = ichunked(dataset_folder, 256)

    _preprocess = partial(
        _preprocess_pipeline,
        destination_folder=DESTINATION_FOLDER
    )

    for batch in batches:
        gc.freeze()
        with futures.ProcessPoolExecutor() as executor:
            executor.map(_preprocess, batch)

if __name__ == '__main__':
    DATASET_FOLDER = '/mnt/hdd_raid/datasets/VGGFace2/train/'
    DESTINATION_FOLDER = '/mnt/hdd_raid/datasets/VGGFace2_Aligned/train/'
    #DATASET_FOLDER = '/mnt/hdd_raid/datasets/TESTE/t1/'
    #DESTINATION_FOLDER = '/mnt/hdd_raid/datasets/TESTE/t2/'

    detect_and_align_faces(DATASET_FOLDER, DESTINATION_FOLDER)
