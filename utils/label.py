import os
import numpy as np
import cv2

class Label():
    def __init__(self, p_id,p_dir,label_unformatted : str):
        self.p_id = p_id

        label_array_format = label_unformatted.split(" ")
        
        self.img_path = os.path.join(p_dir,label_array_format[0])
        self.gaze_target = np.array(label_array_format[24:27],float)

        self.face_center = np.array(label_array_format[21:24],float)
        self.r_vec = np.array(label_array_format[15:18],float)
        self.t_vec = np.array(label_array_format[18:21],float)
        self.rot =  cv2.Rodrigues(self.r_vec)[0]