# Gaze Estimation Official implementation - Author: Enzo Turletti - Francisco Santarelli.

import torch.nn as nn
import timm
import torch
class myModel(nn.Module):
    def __init__(self,eth_xgaze_pretrain = False, eth_xgaze_path = None):
        super(myModel, self).__init__()

        self.model_resnet18 = timm.create_model("resnet18", num_classes=2)

        if eth_xgaze_pretrain:
            print("Using ETH-XGAZE pretrained model as backbone.")
            checkpoint = torch.load(eth_xgaze_path,map_location='cpu')    
            self.model_resnet18.load_state_dict(checkpoint["model"])

    def forward(self, x):
        x = self.model_resnet18(x)

        pitch = x[:,0]
        yaw   = x[:,1]

        return yaw, pitch

