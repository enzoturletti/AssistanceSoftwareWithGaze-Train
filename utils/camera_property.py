import dataclasses
import numpy as np
import yaml


@dataclasses.dataclass()
class CameraProperties:
    width: int = dataclasses.field(init=False)
    height: int = dataclasses.field(init=False)
    camera_matrix: np.ndarray = dataclasses.field(init=False)
    dist_coefficients: np.ndarray = dataclasses.field(init=False)

    camera_params_path: dataclasses.InitVar[str] = None

    def __post_init__(self, camera_params_path):
        with open(camera_params_path) as f:
            data = yaml.safe_load(f)
        self.width = data['image_width']
        self.height = data['image_height']
        self.camera_matrix = np.array(data['camera_matrix']['data']).reshape(3, 3)
        self.dist_coefficients = np.array(data['distortion_coefficients']['data']).reshape(-1, 1)