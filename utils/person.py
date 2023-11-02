import os 
import scipy.io as sio

from utils.config_instance import ConfigInstance

class Person():
    def __init__(self, id, dir_path):
        config_instance = ConfigInstance()
        
        self.id = id
        self.dir_path = dir_path
        self.label_path = os.path.join(self.dir_path,self.id) + ".txt"
        self.dist_coeffs = sio.loadmat(os.path.join(self.dir_path, "Calibration", "Camera.mat"))["distCoeffs"]
        self.camera_matrix = sio.loadmat(os.path.join(self.dir_path, "Calibration", "Camera.mat"))["cameraMatrix"]
        self.length = self.calculate_length()

        # Folders creations.
        output_preprocess_proccess = config_instance.output_dataset_preproccess

        output_preprocess_proccess_images = os.path.join(output_preprocess_proccess, "normalizated_images")
        output_preprocess_proccess_labels = os.path.join(output_preprocess_proccess, "labels")

        if not os.path.exists(output_preprocess_proccess_images):
            os.makedirs(output_preprocess_proccess_images)
        if not os.path.exists(output_preprocess_proccess_labels):
            os.makedirs(output_preprocess_proccess_labels)

        self.images_normalizated_path = os.path.join(output_preprocess_proccess_images,self.id)
        self.label_normalizated_path  = os.path.join(output_preprocess_proccess_labels,self.id+".label")

        if not os.path.exists(self.images_normalizated_path):
            os.makedirs(self.images_normalizated_path)
        ####################

    def calculate_length(self):
        with open(os.path.join(self.dir_path,self.id) + ".txt", 'r') as f:
            lines = f.readlines()
            length = len(lines)
        return length

