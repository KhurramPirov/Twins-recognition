# -*- coding: utf-8 -*-
# Roman Kiryanov, roman.kiryanov@skoltech.ru

# IMPORTS
# !pip install dlib
# !pip install opencv-python
# !pip install --upgrade imutils
# !pip install git+https://github.com/siriusdemon/pytorch-pcn

import dlib
import cv2
import numpy as np
import imutils
import os
from matplotlib import pyplot as plt


# Classes
class dlib_landmark_detector():
    """
    dlib-based face detector
    Initiation takes 3 parameters:
    >> cnn_flag
    If True - use the bounding_box detector based on CNN. The detector will be
    loaded from bb_detector_path, it also changes the way of shape convertation

    >> bb_detector_path
    Path to bounding_box detector based on CNN from dlib

    >> landmark_predictor_path
    Path to frontal face landmark predictor from dlib. Initially it is 68-points,
    but can also be modified to use 5-points

    >> verbose
    Flag for printing the messages

    Functions:
    >> self.image_prepare(image)
    Image rescaling and creating a grayscaled copy for the bb_detector
    it is recommended in dlib documentation to rescale to 500px in width.
    Also dlib bounding box detectors are not working on RGB images

    >> self.covert_rectangles_to_cv_bb()
    Supplementary function for convertation of rectangles from dlib to numpy coordinates

    >> self.covert_predictor_shape_to_np(shape, num_of_landmarks=68, dtype="int")
    Supplementary function, converts predictor shapes into numpy coordinate arrays
    num_of_landmarks is a type of detected landmarks, by default we use 68-points

    >> self.face_detect(image, face_number=0, visualize=False)
    Detecting face in 2 steps: bb and landmarks.
    face number is a number of faces, if there is more then one detected.
    if visualize == True, then visualization of steps on matplotlib subplots

    >> self.visualize()
    Function for plotting the resulting images

    >> self.get_original_size_landmarks()
    Function tt get the coordinates of landmarks on the original size image

    >> self.reset()
    Function to reset all values after prediction

    """

    def __init__(self, cnn_flag, bb_detector_path=None, landmark_predictor_path=None, verbose=False):
        # Check all necessary modules
        # flags
        self.cnn_flag = cnn_flag
        self.verbose_flag = verbose

        # initialize bounding_box_predictor
        if self.cnn_flag:
            if os.path.exists(bb_detector_path):
                self.bb_detector = dlib.cnn_face_detection_model_v1(bb_detector_path)
            else:
                print(
                    'Warning! CNN flag was set TRUE, but the provided bb_detector_path was not correct (file or folder not found)')
                print('Bounding Box detector set to default dlib.get_frontal_face_detector()')
                self.bb_detector = dlib.get_frontal_face_detector()
        else:
            self.bb_detector = dlib.get_frontal_face_detector()

        # Save path for switching model
        self._bb_detector_path = bb_detector_path

        # initialize landmark predictor
        if os.path.exists(landmark_predictor_path):
            self.predictor = dlib.shape_predictor(landmark_predictor_path)
        else:
            print('Warning! The provided path to landmark_predictor was not correct  (file or folder not found)')

        # number of landmarks detected by default shape predictor
        if landmark_predictor_path[-21:-19] == '68':
            self.landmarks_type = 68
        else:
            self.landmarks_type = 68  # by default we use 68-based-points

        # Initialization of inner variables
        self.image_original_shape = None
        self._rescaled_image = None
        self._grayscaled_image = None
        self.scaling_factor = 1

        self.cv_bb_tuple = None
        self.landmark_shape = None
        self.image_dots = None

    # ---------------------------------------------------------------------------

    def image_prepare(self, image):
        """
        Image rescaling and creating a grayscaled copy for the bb_detector
        """
        # 1. Rescale (it is recommended in dlib documentation to rescale to 500px in width)
        new_width = 500
        self.image_original_shape = image.shape

        self.scaling_factor = new_width / image.shape[1]
        self._rescaled_image = imutils.resize(image, width=new_width)

        # 2. convert to grayscale
        self._grayscaled_image = cv2.cvtColor(self._rescaled_image, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------------------------------

    def covert_rectangles_to_cv_bb(self, rectangle):
        """
        take a bounding predicted by dlib and convert it to
        the format (x, y, w, h) as we would normally do with OpenCV
        """
        x = rectangle.left()
        y = rectangle.top()
        w = rectangle.right() - x
        h = rectangle.bottom() - y
        return (x, y, w, h)

    # ---------------------------------------------------------------------------

    def covert_predictor_shape_to_np(self, shape, num_of_landmarks=68, dtype="int"):
        """
        Converts predictor shapes into numpy coordinate arrays
        num_of_landmarks is a type of detected landmarks, by default we use 68
        """
        # initialize the list of (x, y)-coordinates
        numpy_coords = np.zeros((68, 2), dtype=dtype)

        # loop over facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, num_of_landmarks):
            numpy_coords[i] = (shape.part(i).x, shape.part(i).y)

        # return the list of (x, y)-coordinates
        return numpy_coords

    # ---------------------------------------------------------------------------

    def face_detect(self, image, face_number=0, visualize=False):
        """
        Detection of face in 2 steps:
        1. Bounding box detector
        2. Landmark predictor
        """

        # Detection with boundary box
        if self._grayscaled_image is None:
            self.image_prepare(image)

        rectangles = self.bb_detector(self._grayscaled_image, 1)

        # If we detected some faces
        if (len(rectangles) != 0) and (face_number < len(rectangles)):
            if self.cnn_flag:
                face_rectangle = rectangles[face_number].rect
            else:
                face_rectangle = rectangles[face_number]

            if self.verbose_flag:
                print('dlib.face_detect() info: dlib bounding box detector detected', len(rectangles), 'face(s)')
        else:
            face_rectangle = None
            if self.verbose_flag:
                print(
                    'dlib.face_detect() WARNING: dlib bounding box detector did not detected enough faces at given image')

        # Prediction of face landmarks
        if face_rectangle is not None:
            self.cv_bb_tuple = self.covert_rectangles_to_cv_bb(face_rectangle)

            # Face landmark prediction
            self.landmark_shape = self.predictor(self._rescaled_image, face_rectangle)

            if visualize:
                self.image_dots = self.visualize(plot=True)
        else:
            self.cv_bb_tuple = None
            self.landmark_shape = None

    # ---------------------------------------------------------------------------

    def visualize(self, plot=True):
        """
        function for plotting the resulting images
        """
        # Get bounding box coordinates in appropriate for CV2 form
        if self.cv_bb_tuple is not None:
            (x, y, w, h) = self.cv_bb_tuple

            # Visualize bounding box (bb)
            image_bb = cv2.rectangle(self._rescaled_image.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)
            image_bb = cv2.cvtColor(image_bb, cv2.COLOR_BGR2RGB)
            image_bb = cv2.cvtColor(image_bb, cv2.COLOR_BGR2RGB)

            # Visualize landmark dots
            shape = self.covert_predictor_shape_to_np(self.landmark_shape, num_of_landmarks=self.landmarks_type)
            image_dots = image_bb.copy()

            for (x, y) in shape:
                image_dots = cv2.circle(image_dots, (x, y), 3, (0, 0, 255), -1)

            # Face crop
            face = dlib.get_face_chip(self._rescaled_image.copy(), self.landmark_shape)

            # Matplotlib visualization
            if plot:
                plt.figure(figsize=(20, 8))
                plt.subplot(131, xticks=(()), yticks=(()), title='Bounding Box')
                plt.imshow(image_bb)

                plt.subplot(132, xticks=(()), yticks=(()), title='Face Landmarks')
                plt.imshow(image_dots)

                plt.subplot(133, xticks=(()), yticks=(()), title='Face Cropped')
                plt.imshow(face)
                plt.show()

            return imutils.resize(image_dots, width=self.image_original_shape[1])
        else:
            return None

    # ---------------------------------------------------------------------------

    def get_original_size_landmarks(self):
        """
        Function tt get the coordinates of landmarks on the original size image
        """
        if self.landmark_shape is not None:
            shape = self.covert_predictor_shape_to_np(self.landmark_shape, num_of_landmarks=self.landmarks_type)
            return (shape / self.scaling_factor).astype(int)
        else:
            if self.verbose_flag:
                print(
                    'dlib.get_original_size_landmarks() WARNING: No face were detected, check or re-use self.face_detect() function')
            return None

    # ---------------------------------------------------------------------------

    def reset(self):
        """
        Function to reset all values after prediction
        """
        self.image = None
        self._rescaled_image = None
        self._grayscaled_image = None
        self.scaling_factor = 1

        self.cv_bb_tuple = None
        self.landmark_shape = None

    # ---------------------------------------------------------------------------

    def switch_model(self):
        """
        Function for switching model between HOG based and CNN based,
        if last is awailable
        """
        # initialize bounding_box_predictor
        if self.cnn_flag:
            # Was true: CNN -> HOG
            self.bb_detector = dlib.get_frontal_face_detector()
        else:
            # Was false: HOG -> CNN if awailable
            if os.path.exists(self._bb_detector_path):
                self.bb_detector = dlib.cnn_face_detection_model_v1(self._bb_detector_path)
            else:
                print(
                    'Warning! CNN flag was set TRUE, but the provided bb_detector_path was not correct (file or folder not found)')
                print('Boundig Box detector set to default dlib.get_frontal_face_detector()')
                self.bb_detector = dlib.get_frontal_face_detector()

        self.cnn_flag = not self.cnn_flag
        self.reset()
        # ---------------------------------------------------------------------------
