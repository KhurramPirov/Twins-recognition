# -*- coding: utf-8 -*-
# Roman Kiryanov, roman.kiryanov@skoltech.ru

# IMPORTS
# !pip install dlib
# !pip install opencv-python
# !pip install --upgrade imutils
# !pip install git+https://github.com/siriusdemon/pytorch-pcn

import cv2
import numpy as np

# Classes
class head_rotation_estimator():
    """
    Estimator of head rotation angles, based on detected original size landmarks
    Initiation takes 1 parameter:
    >> np_shape
    It is a numpy ndarray of 68 facial landmarks, detected by dlib predictors

    Functions:
    >> angle_eye, angle_nose = self.hor_distance_to_angle(eye_distance, nose_distance)
    Supplementary function for converting relative distances into angles of
    head rotation, based on investigation of several selfies

    >> angle_nose_mouth, angle_mouth_jaw = ver_distance_to_angle(eye_eyebrow_distance, nose_mouth_distance, mouth_jaw_distance)
    Supplementary function for converting relative distances into angles of
    head rotation, based on investigation of several selfies


    >> self.get_horizontal_rotation_angles()
    Function to estimate horizontal angles from face shape landmarks
    """

    def __init__(self, np_shape, verbose=False):
        self.shape = np_shape  # numpy-like shape

        self.horizontal_angle = 0
        self.horizontal_direction = 'front'

        self.vertical_angle = 0
        self.vertical_direction = 'front'

        self.verbose_flag = verbose

    # ---------------------------------------------------------------------------

    def hor_distance_to_angle(self, eye_distance, nose_distance):
        """
        Supplementary function for converting relative distances into angles of
        head rotation, based on investigation of several selfies
        """
        basic_eye_coeff, basic_nose_coeff = 140, 55
        corr_eye_coeff, corr_nose_coeff = 70, 10

        angle_eye = eye_distance * (corr_eye_coeff * eye_distance ** 2 + basic_eye_coeff)
        angle_nose = nose_distance * (corr_nose_coeff * nose_distance ** 2 + basic_nose_coeff)
        return angle_eye, angle_nose

    # ---------------------------------------------------------------------------

    def ver_distance_to_angle(self, r_eye_distance, l_eye_distance):
        """
        Supplementary function for converting relative distances into angles of
        head rotation, based on investigation of several selfies
        """
        av_dist = (r_eye_distance + l_eye_distance) / 2
        basic_coeff, corr_coeff = -0.0008, 0.0075
        vert_angle = (av_dist + basic_coeff) / corr_coeff

        # n1,n2, n3,n4 = -3205.6, 53.087, 161.33, -0.4118
        # vert_angle = n1*np.power(av_dist,3)+n2*np.power(av_dist,2)+n3*av_dist+n4
        return vert_angle

    # ---------------------------------------------------------------------------

    def get_horizontal_rotation_angles(self):
        """
        Function to estimate horizontal angles from face shape landmarks
        """
        # Chech shape
        assert len(self.shape) == 68, "Found shape from face_predictor has not enough points. Please, check it"

        # Decision angles
        angle_front = 10

        right_eye_side = self.shape[:2].mean(axis=0).astype(int)
        left_eye_side = self.shape[15:17].mean(axis=0).astype(int)
        face_width = np.sqrt(np.power(right_eye_side - left_eye_side, 2).sum())

        # Get distances between eye and side
        # right eye right-most part = point 37
        # left eye left-most part = point 46
        right_eye_distance = np.sqrt(np.power(self.shape[36] - right_eye_side, 2).sum()) / face_width
        left_eye_distance = np.sqrt(np.power(self.shape[45] - left_eye_side, 2).sum()) / face_width

        # Get distances between nose point (point 30) and cheek
        right_nose_side = self.shape[1:3].mean(axis=0).astype(int)
        left_nose_side = self.shape[14:16].mean(axis=0).astype(int)
        right_nose_distance = np.sqrt(np.power(self.shape[30] - right_nose_side, 2).sum()) / face_width
        left_nose_distance = np.sqrt(np.power(self.shape[30] - left_nose_side, 2).sum()) / face_width

        # Estimate face rotation via relative distances
        # >0 - rotation to the left
        # <0 - rotation to the right
        face_rotation_eye = right_eye_distance - left_eye_distance
        face_rotation_nose = right_nose_distance - left_nose_distance

        # Angle averaging
        angle_eye, angle_nose = self.hor_distance_to_angle(face_rotation_eye, face_rotation_nose)

        if angle_nose < 9:
            hor_angle = (0.9 * angle_nose + 0.4 * angle_eye) / 2
        else:
            hor_angle = (angle_nose + angle_eye) / 2

        if self.verbose_flag:
            print('hre.get_horizontal_rotation_angles() info:')
            print('   distance: right eye-ear = %.3f' % (right_eye_distance))
            print('   distance: right nose-cheek = %.3f' % (right_nose_distance))
            print('   distance: left eye-ear = %.3f' % (left_eye_distance))
            print('   distance: left nose-cheek = %.3f' % (left_nose_distance))
            print('   diff distances: eye-ear = %.3f, nose-cheek = %.3f' % (face_rotation_eye, face_rotation_nose))
            print('   angles: eye-ear = %.3f, nose-cheek = %.3f' % (angle_eye, angle_nose))
            print('----------')

        # Direction decision
        if abs(hor_angle) <= angle_front:
            self.horizontal_angle = int(round(hor_angle))
            self.horizontal_direction = 'front'
        elif hor_angle < 0:
            self.horizontal_angle = int(round(hor_angle))
            self.horizontal_direction = 'right'
        else:
            self.horizontal_angle = int(round(hor_angle))
            self.horizontal_direction = 'left'

    # ---------------------------------------------------------------------------

    def get_vertical_angles(self):
        """
        Function to estimate vertical angles from face shape landmarks
        Based on: https://studref.com/323754/pravo/prilozhenie_opredelenie_polozheniya_golovy_sfotografirovannogo_litsa
        """
        # Shape check
        assert len(self.shape) == 68, "Found shape from face_predictor has not enough points. Please, check it"

        # Decision angles
        angle_vertical = 5

        # Get face height
        down_jaw = self.shape[7:10].mean(axis=0).astype(int)
        up_right_eyebrow = self.shape[18:21].mean(axis=0).astype(int)
        up_left_eyebrow = self.shape[23:26].mean(axis=0).astype(int)
        up_middle = (up_right_eyebrow + up_left_eyebrow) / 2
        face_height = np.sqrt(np.power(down_jaw - up_middle, 2).sum())

        # Get down side of eye
        down_right_eye = self.shape[40:42].mean(axis=0).astype(int)
        down_left_eye = self.shape[46:48].mean(axis=0).astype(int)

        # Get ears points for building 'line'
        p1, p2 = self.shape[0], self.shape[16]

        # Get distances from down side of eyes to line
        p3 = down_left_eye
        distance_le = np.cross(p2 - p1, p1 - p3) / np.linalg.norm(p2 - p1, 2)
        distance_le = distance_le / face_height

        p3 = down_right_eye
        distance_re = np.cross(p2 - p1, p1 - p3) / np.linalg.norm(p2 - p1, 2)
        distance_re = distance_re / face_height

        # Get angle
        aver_angle = self.ver_distance_to_angle(distance_re, distance_le)

        if self.verbose_flag:
            print('hre.get_vertical_rotation_angles() info:')
            print('   distance: left  eye to line = %.3f' % (distance_le))
            print('   distance: right eye to line  = %.3f' % (distance_re))
            print('   angle = %.3f' % (aver_angle))
            print('----------')

        if abs(aver_angle) <= angle_vertical:
            self.vertical_angle = int(round(aver_angle))
            self.vertical_direction = 'front'
        elif aver_angle < 0:
            self.vertical_angle = int(round(aver_angle))
            self.vertical_direction = 'down'
        else:
            self.vertical_angle = int(round(aver_angle))
            self.vertical_direction = 'up'

    # ---------------------------------------------------------------------------

    def visualize(self, original_size_image, plot=True, plus_radius=5):
        """
        Function for visuzualization
        """
        # down part of eye
        image = original_size_image.copy()

        down_right_eye = self.shape[40:42].mean(axis=0).astype(int)
        down_left_eye = self.shape[46:48].mean(axis=0).astype(int)

        parts = [self.shape[0], self.shape[16],  # side parts
                 down_right_eye, down_left_eye,  # down sides of eyes
                 ]
        for (x, y) in parts:
            image = cv2.circle(image, (x, y), 3 + plus_radius, (255, 0, 0), -1)

        # Draw line between ears
        lineThickness = 2
        cv2.line(image, (parts[0][0], parts[0][1]), (parts[1][0], parts[1][1]), (255, 0, 0), lineThickness)

        return image