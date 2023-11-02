import configparser
import math
class ConfigInstance:
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance
    
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('config.ini')

        self.original_dataset_path = config.get('paths', 'original_dataset_path')
        self.output_dataset_preproccess = config.get('paths', 'output_dataset_preproccess')
        self.normalizated_distance_m = float(config.get("normalizated_camera","normalizated_distance"))
        self.normalizated_distance_mm = self.normalizated_distance_m*1000
        self.normalizated_camera_intrinsic_path = config.get('paths', 'normalized_camera_intrinsic')
