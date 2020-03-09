import os
import psutil
import multiprocessing as mp
from mtcnn import MTCNN


class Watcher():
    def __init__(self):
        self._manager = mp.Manager()
        self._generate_queues()
        self._face_detector = MtcnnDetectFaces(
            self._images_queue,
            self._detected_faces_queue,
            self._process_intercommunication_queue
        )
        self._face_detector.attach(self)

    def _generate_queues(self):
        self._images_queue = self._manager.Queue()
        self._detected_faces_queue = self._manager.Queue()
        self._process_intercommunication_queue = self._manager.Queue()

    @property
    def images_queue(self):
        return self._images_queue

    @property
    def detected_faces_queue(self):
        return self._detected_faces_queue

    @property
    def process_intercommunication_queue(self):
        return self._process_intercommunication_queue

    def run(self):
        self._face_detector.run()

    def update(self, process_join=False):
        process_join()
        self._face_detector.run()

    def stop(self):
        self._face_detector.stop()

class MtcnnDetectFaces():
    def __init__(
            self,
            images_queue,
            detected_faces_queue,
            process_intercommunication_queue
        ):
        self._observer = None
        self._process = None
        self._images_queue = images_queue
        self._detected_faces_queue = detected_faces_queue
        self._process_intercommunication_queue = process_intercommunication_queue

    def attach(self, observer):
        self._observer = observer

    def _notify(self):
        self._observer.update(self._process.join)

    def _memory_usage(self):
        pid = os.getpid()
        process = psutil.Process(pid)
        mem = process.memory_info().rss / float(2 ** 20)
        print(f'PID: {pid} -- Mem: {mem}')
        return mem

    def _detect_faces(self):
        # Imports TF everytime, necessary for multiprocessing the pipeline
        print('_mtcnn_detect_faces started')
        import tensorflow as tf
        gpus = tf.config.experimental.get_visible_devices()
        tf.config.experimental.set_memory_growth(gpus[-2], True)
        tf.config.experimental.set_memory_growth(gpus[-1], True)

        while True:
            if self._stopping_condition():
                break
            if not self._images_queue.empty():
                image_container = self._images_queue.get(True, 5)
                detected_face = MTCNN().detect_faces(image_container.image)
                self._detected_faces_queue.put((image_container, detected_face))
        print('_mtcnn_detect_faces Done!')

    def _stopping_condition(self):
        if self._memory_usage() >= 3_000:
            self._notify()
            return True
        if not self._process_intercommunication_queue.empty():
            if self._process_intercommunication_queue.get() == 'Stop':
                return True

    def run(self):
        self._process = mp.Process(
            target=self._detect_faces
        )
        self._process.start()

    def stop(self):
        self._process_intercommunication_queue.put('Stop')
        self._process.join()

if __name__ == '__main__':
    mp.set_start_method('spawn')
