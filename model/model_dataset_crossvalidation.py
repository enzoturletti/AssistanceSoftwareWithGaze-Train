import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from sklearn.model_selection import train_test_split

class GazeDataset(Dataset):
    def __init__(self, rootpath, train : bool, retire_pids = []):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        labels_path = os.path.join(rootpath, "output_preproccess", "labels")
        normalizated_imgs_path = os.path.join(rootpath, "output_preproccess", "normalizated_images")
        person_ids = os.listdir(normalizated_imgs_path)
        person_ids.sort()
        
        # Filter persons.
        if train:
            person_ids = [x for x in person_ids if x not in retire_pids]
            print("Train person ids",person_ids)

        else:
            person_ids = retire_pids
            print("Eval person ids",person_ids)

        samples = []
        for person_id in person_ids:
            label_path = os.path.join(labels_path, person_id + ".label")

            with open(label_path, "r") as label_file:
                for i, line in enumerate(label_file):
                    if i == 0:
                        continue

                    fields = line.strip().split()

                    img_path = os.path.join(rootpath, normalizated_imgs_path, person_id, os.path.basename(fields[0]))
                    yaw_angle = float(fields[1])
                    pitch_angle = float(fields[2])

                    samples.append((img_path, yaw_angle, pitch_angle))
                    
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        norm_img = Image.open(self.samples[idx][0])
        yaw_angle = self.samples[idx][1]
        pitch_angle = self.samples[idx][2]

        norm_img = self.transform(norm_img)

        return norm_img, yaw_angle, pitch_angle


