import numpy as np

class GazeVectorUtils():
    @staticmethod
    def normalizated_3d_gaze_vector(gaze_vector, conversion_matrix):
        normalizated_gaze_3d = conversion_matrix @ gaze_vector
        return normalizated_gaze_3d / np.linalg.norm(normalizated_gaze_3d)

    @staticmethod
    def denormalizated_3d_gaze_vector(normalizated_gaze_vector, conversion_matrix):
        denormalizated_gaze_vector = np.linalg.inv(
            conversion_matrix) @ normalizated_gaze_vector
        return (denormalizated_gaze_vector/np.linalg.norm(denormalizated_gaze_vector)).reshape(1,3)[0]

    @staticmethod
    def gaze3d_to_gaze2d(gaze_vector_3d):
        yaw = np.arctan2(-gaze_vector_3d[0], -gaze_vector_3d[2])
        pitch = np.arcsin(-gaze_vector_3d[1])
        return np.array([yaw, pitch])

    @staticmethod
    def gaze2d_to_gaze3d(gaze_vector_2d):
        x = -np.cos(gaze_vector_2d[1]) * np.sin(gaze_vector_2d[0])
        y = -np.sin(gaze_vector_2d[1])
        z = -np.cos(gaze_vector_2d[1]) * np.cos(gaze_vector_2d[0])
        return np.array([x, y, z])