import cv2
#import tensorflow as tf
import numpy as np
from mtcnn import MTCNN
from skimage import transform
# Ver problema do @tf.functions

def _align_faces(
        image,
        facial_landmarks,
        bounding_box=None,
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
    assert crop_shape == (112, 112) or (112, 96)
    if crop_shape == (112, 96): 
        crop_shape = (96, 112)

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

    #transformation_matrix = None
    #if transformation_matrix is None: # Align using bounding_box center
    #    return tf.image.crop_to_bounding_box(
    #        image,
    #        bounding_box[1],
    #        bounding_box[0],
    #        bounding_box[3],
    #        bounding_box[2]
    #    )
    #else: # Align using facial_landmarks
    try:
        return cv2.warpAffine(
            image,
            transformation_matrix,
            crop_shape,
            borderValue=0.0
        )
    except Exception:
        print('An error occurred, probably transformation_matrix is None')

def _extract_center_face(image, result):
    """Extracts the closest face of the center of a given image.

    ## Parameters
        image: image with the faces.
        result: list of resulting bounding boxes and keypoints from
        MTCNN().detect_faces().

    ## Returns
        (bounding_box, facial_landmarks) for the closest face to the center of
        the image.
    """
    center = np.asarray([image.shape[0] / 2, image.shape[1] / 2])
    center_face = {}

    for face in result:
        mean_bbox = np.asarray([
            np.mean([face['box'][0], face['box'][1]]),
            np.mean([face['box'][2], face['box'][3]])
        ])
        distance_from_center = np.linalg.norm(mean_bbox - center)
        center_face[distance_from_center] = face

    minimum_distance = min(center_face.keys())
    bounding_box = center_face[minimum_distance]['box']
    facial_landmarks = center_face[minimum_distance]['keypoints']

    return  bounding_box, facial_landmarks

def _detect_faces(image):
    """Detects faces from a given image.

    ## Parameters
        image: the image to be searched on.

    ## Returns
        (bounding_box, facial_landmarks) for the face.
    """
    result = MTCNN().detect_faces(image)

    if len(result) > 1:
        bounding_box, keypoints = _extract_center_face(image, result)
    else:
        bounding_box = result[0]['box']
        keypoints = result[0]['keypoints']

    facial_landmarks = [
        keypoints['left_eye'],
        keypoints['right_eye'],
        keypoints['nose'],
        keypoints['mouth_left'],
        keypoints['mouth_right']
    ]

    return bounding_box, facial_landmarks

def detect_and_align_faces(images, crop_shape=(122, 122)):
    """Detect faces in a given image using MTCNN, align them and crops to a
    given size.

    ## Parameters
        image: image to have faces detected.
        crop_shape: optional shape of the output image.

    ## Returns
        cropped and align face.
    """
    bounding_box, facial_landmarks = _detect_faces(images)

    return _align_faces(images, facial_landmarks, bounding_box, crop_shape)



#image = cv2.cvtColor(cv2.imread("test2.jpg"), cv2.COLOR_BGR2RGB)
##image = tf.image.decode_jpeg(image)
##image = tf.image.convert_image_dtype(image, tf.float32)
#image = detect_and_align_faces(image)
#image = tf.io.encode_jpeg(image)
#tf.io.write_file('image_.jpg', image)
