import multiprocessing as mp
from mtcnn import MTCNN


class Watcher():
    def __init__(self):
        self._manager = mp.Manager()
        self._generate_queues()
        #self._create_face_detector_process()
        self._face_detector_process = None
        self._watcher_process = None

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

    def create_face_detector_process(self):
        self._watcher_process = mp.Process(
            target=self._create_face_detector_process
        )
        self._watcher_process.start()

    def kill_app(self):
        self._process_intercommunication_queue.put('Done')
        self._watcher_process.join()

    def _create_face_detector_process(self):
        print('_create_face_detector_process')
        self._face_detector_process = mp.Process(
            target=self._mtcnn_detect_faces
        )
        self._face_detector_process.start()

        while True:
            if not self._process_intercommunication_queue.empty():
                if self._process_intercommunication_queue.get() == 'Done':
                    self._process_intercommunication_queue.put('Stop')
                    break
                else:
                    continue
            if self._memory_usage(self._face_detector_process.pid) > 8_000:
                self._process_intercommunication_queue.put('Stop')
                self._face_detector_process = mp.Process(
                    target=self._mtcnn_detect_faces
                )
                self._face_detector_process.start()
        self._face_detector_process.join()
        print('_create_face_detector_process Done!')

    def _memory_usage(self, pid):
        # return the memory usage in MB
        import psutil
        process = psutil.Process(pid)
        mem = process.memory_info().rss / float(2 ** 20)
        print(mem)
        return mem

    def _mtcnn_detect_faces(self):
        # Imports TF everytime, necessary for multiprocessing the pipeline
        print('=' * 30)
        print('_mtcnn_detect_faces started')
        import tensorflow as tf
        gpus = tf.config.experimental.get_visible_devices()
        tf.config.experimental.set_memory_growth(gpus[-2], True)
        tf.config.experimental.set_memory_growth(gpus[-1], True)

        while True:
            if not self._process_intercommunication_queue.empty():
                if self._process_intercommunication_queue.get() == 'Stop':
                    break
                else:
                    continue
            if not self._images_queue.empty():
                image_container = self._images_queue.get(True, 5)
                detected_face = MTCNN().detect_faces(image_container.image)
                self._detected_faces_queue.put((image_container, detected_face))
        print('-' * 30)
        print('_mtcnn_detect_faces Done!')

class GpuProcess():
    def __init__(self, observer):
        self._observer = None

    def attach(self, observer):
        self._observer = observer

    def notify(self):
        self._observer.update()

if __name__ == '__main__':
    mp.set_start_method('spawn')