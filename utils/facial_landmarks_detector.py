import numpy as np
import mediapipe as mp

class FaceLandmarksDetector:
    def __init__(self, min_detection_confidence : float = 0.5, min_tracking_confidence : float = 0.5):
        self.face_mesh =  mp.solutions.face_mesh.FaceMesh(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)

    def detect(self,image):
        height,width,_ = image.shape

        results = self.face_mesh.process(image)
        if(results.multi_face_landmarks != None):
            face_landmarks = results.multi_face_landmarks[0]
            landmarks2d = np.array([(lm.x*width, lm.y*height) for lm in face_landmarks.landmark],dtype=float)

            return True, landmarks2d
        else:
            return False, False