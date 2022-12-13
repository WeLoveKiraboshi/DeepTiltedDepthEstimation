import sys
import cv2
import numpy as np


class VideoWriter(object):
    def __init__(self, save_path='./video.mp4', fps=25, imsize=(224, 224)):
        # encoder(for mp4)
        #imsize must be tuplle object
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        # output file name, encoder, fps, size(fit to image size)
        self._check_np = lambda x: isinstance(x, np.ndarray)
        self.save_path = save_path
        self.video = cv2.VideoWriter(self.save_path, fourcc, fps, imsize)
        if not self.video.isOpened():
            print("can't be opened video in utils.VideoWriter.py")
            sys.exit()
        if not '.mp4' in self.save_path:
            print('indicated save path is incorrect. {}'.format(self.save_path))
            exit(0)

    def add_frame(self, img):
        if img is None:
            print("can't read frame : frame is None object")
            exit(0)
        if self._check_np(img):
            self.video.write(img)
        else:
            print('img object must be numpy ndarray. please check type again...')
            exit(0)
        pass

    def finish(self,):
        self.video.release()
        print('finish writing in VideoWriter. save_path = {}'.format(self.save_path))
