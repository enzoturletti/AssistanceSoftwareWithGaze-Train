import cv2
from utils.camera_property import CameraProperties
from utils.facial_landmarks_detector import FaceLandmarksDetector
from utils.facial_landmarks_3d_detector import FacialLandmarks3dDetector
from utils.face_model import FaceModel
from utils.config_instance import ConfigInstance
import numpy as np

class FrameNormalizator:

    def __init__(self):
        config_instance = ConfigInstance()
        self.norm_camera_properties = CameraProperties(config_instance.normalizated_camera_intrinsic_path)

        self.face_landmarks_detector = FaceLandmarksDetector()
        self.face_landmarks_3d_detector = FacialLandmarks3dDetector()

        self.normalizated_image = None
        self.R_mat = None
        self.face_center = None
        self.rot = None
        self.tvec = None
        self.facial_landmarks_3d = None

    def run_image_normalization(self, face_image, camera_matrix, dist_coeffs, norm_dist):

        # 2D Facial Landmarks detection.
        success, facial_landmarks_2d = self.face_landmarks_detector.detect(face_image)
        if not success:
            return False
        
        # Rotation and translation vector calculation.
        rot, tvec = self.calculate_rot_tvec(facial_landmarks_2d, FaceModel().get(), camera_matrix, dist_coeffs)

        # Face center calculation
        facial_landmarks_3d, face_model = self.face_landmarks_3d_detector.estimate_with_face_model(rot, tvec)
        face_center = self.calculate_face_center(face_model)

        # Image normalization
        normalizated_image, R_mat = self.calculate_normalizated_image(face_image, rot, face_center,camera_matrix,norm_dist)

        self.rot = rot
        self.tvec = tvec 
        self.facial_landmarks_3d = facial_landmarks_3d
        self.facial_landmarks_2d = facial_landmarks_2d
        self.face_center = face_center
        self.normalizated_image = normalizated_image
        self.R_mat = R_mat


        return True
    
    def calculate_face_center(self, facial_landmarks_3d):
        return facial_landmarks_3d.mean(axis=0)

    def calculate_rot_tvec(self, facial_landmarks_2d, facial_landmarks_3d, camera_matrix, dist_coeffs):

        rvec = np.zeros(3, dtype=float)
        tvec = np.array([0, 0, 1], dtype=float)
        _, rvec, tvec = cv2.solvePnP(facial_landmarks_3d,
                                     facial_landmarks_2d,
                                     camera_matrix,
                                     dist_coeffs,
                                     rvec,
                                     tvec,
                                     useExtrinsicGuess=True,
                                     flags=cv2.SOLVEPNP_ITERATIVE)
        rot = cv2.Rodrigues(rvec)[0]

        return rot, tvec

    def calculate_normalizated_image(self, face_image, rot, face_center,camera_matrix, norm_dist):

        S_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, norm_dist/np.linalg.norm(face_center)]])

        xaxis = rot[:, 0]
        z = face_center / np.linalg.norm(face_center)
        y = np.cross(z, xaxis)
        y = y / np.linalg.norm(y)
        x = np.cross(y, z)
        x = x/np.linalg.norm(x)

        R_mat = np.array([x, y, z])

        C_mat = self.norm_camera_properties.camera_matrix
        M_mat = np.dot(S_mat, R_mat)
        W_mat = np.dot(np.dot(C_mat, M_mat), np.linalg.inv(camera_matrix))

        normalized_image = cv2.warpPerspective(face_image, W_mat, (self.norm_camera_properties.width, self.norm_camera_properties.height))
        
        return normalized_image, R_mat
