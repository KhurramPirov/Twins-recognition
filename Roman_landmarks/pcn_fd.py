# -*- coding: utf-8 -*-
# Roman Kiryanov, roman.kiryanov@skoltech.ru

# IMPORTS
# !pip install dlib
# !pip install opencv-python
# !pip install --upgrade imutils
# !pip install git+https://github.com/siriusdemon/pytorch-pcn

import cv2
import numpy as np
import imutils
import pcn
from matplotlib import pyplot as plt

# Classes
class PCN_face_detector():
    """
    PCN based rotation estimator
    PCN is a Progressive Calibration Network for rotation-invariant face recognition
      It is Pytorch-based trained model: https://arxiv.org/pdf/1804.06039.pdf

    Initiation takes image as 3 (RGB) or 1 (grayscaled) image with 1 (or more) faces
    >> verbose: for printing messages

    Functions:
    >> self.get_bb_pcn()
    Getting bounding box (bb) for all detected face images on several resolutions.

    >> self.angle = get_angle(face_number=0)
    Getting the averaged angles from detected bounding boxes (bb)
    for detected face with num "face_number"

    >> self.restore(visualize=False)
    Rotation of image on angle, detected previously and
    visualizing the result of angle detection
    Draws for every face, detected at the picture.
    Uses matplotlib.pyplot
    """

    def __init__(self, image, verbose=False):
        self.image = image.copy()

        self.faces = []
        if self.image.shape[1] < 1000:
            self.widths = [50, 100, 200, 500, self.image.shape[1]]
        else:
            self.widths = [50, 100, 200, 500]
        self.angles = np.zeros(len(self.widths))
        self.face_angle = 0
        self.__version__ = 0.1

        self.verbose = verbose
        # -----------------------------------------------------------------------------

    def get_bb_pcn(self):
        """
        Function for getting the bounding box (bb) for all
        detected face images on several resolutions

        Input  >> image, (hight, width, channel)
        Output >> None
        """
        for i, width in enumerate(self.widths):
            # Resize to concrete resolution
            resized_image = imutils.resize(self.image.copy(), width=width)
            # Get bounding boxes
            self.faces.append(pcn.detect(resized_image))

    # -----------------------------------------------------------------------------

    def get_angle(self, face_number=0):
        """
        Function for getting the averaged angles from
        detected bounding boxes (bb) for detected face with num "face_number"

        Input  >> face_number, int, by default it is 0
        Output >> angle of face rotation, either angle or 0
        """
        for i, width in enumerate(self.widths):
            if len(self.faces[i]) == 0:
                # No faces detected at all
                self.angles[i] = 0
            else:
                self.angles[i] = self.faces[i][face_number].angle

        nnz = np.nonzero(self.angles)
        if len(self.angles[nnz]) == 0:
            # No face detected at any resolution
            self.face_angle = 0
            if self.verbose:
                print('pcn_detector.get_angle(): Warning! Possibly the face has not been found')
        else:
            self.face_angle = self.angles[nnz].mean()
        return self.face_angle

    # -----------------------------------------------------------------------------

    def restore(self, visualize=False):
        """
        Function for visualizing the result of angle detection
        Draws for every face, detected at the picture at every
        Uses matplotlib.pyplot

        Input  >> image, (hight, width, channel)
        Output >> restored_image
        If visualize >> matplotlib subplots
        """
        if len(self.image.shape) == 3:
            # Color image
            rows, cols, _ = self.image.shape
        else:
            # grayscaled image
            rows, cols = self.image.shape

        if self.face_angle == 0:
            # Check if function was called
            self.get_bb_pcn()
            self.face_angle = self.get_angle(face_number=0)

        M_inverse = cv2.getRotationMatrix2D((cols / 2, rows / 2), - self.face_angle, 1)
        image_restored = cv2.warpAffine(self.image.copy(), M_inverse, (cols, rows))

        if visualize:
            self.visualize(image_restored)
        return image_restored

    # -----------------------------------------------------------------------------

    def visualize(self, image_restored):
        """
        Supplementary function for visualization
        """
        image_bb = self.image.copy()

        # inverse order for looking the highest resolution of face detection
        i, width = -1, self.widths[-1]
        counter = len(self.widths)
        while (len(self.faces[i]) == 0) and (counter != 1):
            i, width = i - 1, self.widths[i - 1]
            counter -= 1

        # Drawing of bounding box
        resized_image_bb = imutils.resize(image_bb, width=width)
        pcn.draw(resized_image_bb, self.faces[i])
        image_bb = imutils.resize(resized_image_bb, width=self.image.shape[1])

        if (self.face_angle == 0) and (counter == 1) and (len(self.faces[i]) == 0):
            image_bb = self.image.copy()
            bb_title = 'No detected Bounding Box'
        else:
            bb_title = 'Detected Bounding Box at scale x' + str(round(width / self.image.shape[1], 3))

        plt.figure(figsize=(15, 8))
        plt.subplot(131, xticks=(()), yticks=(()), title='Given Image')
        plt.imshow(self.image)

        plt.subplot(132, xticks=(()), yticks=(()), title=bb_title)
        plt.imshow(image_bb)

        plt.subplot(133, xticks=(()), yticks=(()),
                    title='Restored image with angle = ' + str(round(-self.face_angle, 3)))
        plt.imshow(image_restored)
        plt.show()