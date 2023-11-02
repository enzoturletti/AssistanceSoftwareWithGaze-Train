#This class use MediaPipe Model to estimate FaceModel

import numpy as np
from utils.face_model import FaceModel

class FacialLandmarks3dDetector:
    def __init__(self):
        self.face_model_3d = FaceModel().get()
        
    def estimate_with_face_model(self, rot, tvec):
        REYE_INDICES: np.ndarray = np.array([33, 133])
        LEYE_INDICES: np.ndarray = np.array([362, 263])
        MOUTH_INDICES: np.ndarray = np.array([185, 409])

        facial_landmarks_3d = self.face_model_3d @ rot.T + tvec
        face_model = facial_landmarks_3d[np.concatenate([REYE_INDICES, LEYE_INDICES, MOUTH_INDICES])]

        return facial_landmarks_3d, face_model
    