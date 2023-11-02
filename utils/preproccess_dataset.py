from utils.config_instance import ConfigInstance
from utils.person import Person
from utils.label import Label
from utils.facial_landmarks_detector import FaceLandmarksDetector
from utils.frame_normalizator import FrameNormalizator
from utils.gaze_vector_utils import GazeVectorUtils

import os
import numpy as np
import cv2

class PreprocessingMPIIFaceGaze():
    def __init__(self):
        config_instance = ConfigInstance()
        self.dataset_path = config_instance.original_dataset_path
        self.normalizated_distance_mm = config_instance.normalizated_distance_mm
        self.normalizated_distance_m = config_instance.normalizated_distance_m
        self.total_images = 0
        self.pendent_images = 0

        # Instances
        self.face_landmarks_detector = FaceLandmarksDetector()
        self.face_normalizator_1 = FrameNormalizator()
        self.face_normalizator_2 = FrameNormalizator()

    def start(self):
        data_persons_id = [d for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d))]
        
        persons = []
        for p_id in data_persons_id:
            data_person_path = os.path.join(self.dataset_path, p_id)
            person = Person(p_id, data_person_path)
            persons.append(person)
        persons = sorted(persons, key=lambda x: x.id)

        self.total_images = 0
        for person in persons:
            self.total_images += person.length
        self.pendent_images = self.total_images

        print("Nº images: ", self.total_images)
        for person in persons:
            self.preprocess_person(person)


    def preprocess_person(self,person : Person):

        # Read labels file.
        with open(os.path.join(person.dir_path,person.id) + ".txt", 'r') as f:
            lines = f.readlines()
            length = len(lines)

        labels = []
        for label_unformatted in lines:
            label = Label(person.id,person.dir_path,label_unformatted)
            labels.append(label)

        outfile = open(person.label_normalizated_path, 'w')
        outfile.write("Face_norm Yaw Pitch\n")

        for i,label in enumerate(labels):
            self.pendent_images -= 1
            print("Proccessing image: ", self.total_images-self.pendent_images, "/" , self.total_images)

            im = cv2.imread(label.img_path)

            # Normalization the img with the dataset label.
            im_norm, unused = self.face_normalizator_1.calculate_normalizated_image(im.copy(),label.rot,label.face_center,person.camera_matrix,self.normalizated_distance_mm)

            # Normalization the img with mediapipe.
            success = self.face_normalizator_2.run_image_normalization(im.copy(),person.camera_matrix, person.dist_coeffs, self.normalizated_distance_m)

            if not success:
                continue

            success = cv2.imwrite(os.path.join(person.images_normalizated_path,str(i)+".jpg"), im_norm.copy())
            
            if not success:
                continue

            # Gaze Vector Normalizated reconstruction.

            gaze_vector = label.gaze_target/1000 - self.face_normalizator_2.face_center
            gaze_vector = gaze_vector / np.linalg.norm(gaze_vector)

            R_mat = self.face_normalizator_2.R_mat
            gaze_vector_3d_norm = GazeVectorUtils.normalizated_3d_gaze_vector(gaze_vector,R_mat)
            gaze_vector_2d_norm = GazeVectorUtils.gaze3d_to_gaze2d(gaze_vector_3d_norm)

            absolut_path_label = os.path.normpath(person.images_normalizated_path)
            absolut_path_components = absolut_path_label.split(os.path.sep)
            path_label = os.path.join(os.path.sep, absolut_path_components[-2], absolut_path_components[-1])

            outfile.write(os.path.join(path_label,str(i)+".jpg") + " " + str(gaze_vector_2d_norm[0]) + " " + str(gaze_vector_2d_norm[1]) + "\n")

        outfile.close()




