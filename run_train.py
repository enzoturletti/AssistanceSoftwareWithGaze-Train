import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.model import myModel
from model.model_dataset import GazeDataset
import matplotlib.pyplot as plt

import numpy as np

def convert_gaze2d_to_gaze3d(gaze):
  gaze_gt = np.zeros([3])
  gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
  gaze_gt[1] = -np.sin(gaze[1])
  gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
  return gaze_gt

def angular_error(yaw_pred, pitch_pred, yaw_true, pitch_true):
    # Calculate 2d vectors
    gaze_2d_pred = [yaw_pred,pitch_pred]
    gaze_2d_true = [yaw_true,pitch_true]

    gaze_3d_pred = convert_gaze2d_to_gaze3d(gaze_2d_pred)
    gaze_3d_true = convert_gaze2d_to_gaze3d(gaze_2d_true)

    # Calculate the dot product between predicted and true gaze vectors
    dot_products = np.sum(gaze_3d_pred * gaze_3d_true)

    # Calculate the angular error in degrees
    angular_error = np.arccos(min(dot_products/(np.linalg.norm(gaze_3d_pred)* np.linalg.norm(gaze_3d_true)), 0.9999999))*180/np.pi

    return angular_error

root_path = os.getcwd()
checkpoints_path = os.path.join(root_path, 'output_train')

learning_rate = 0.0001
batch_size = 64

# Initialize model and optimizer
model = myModel(True,"/workspaces/AssistanceSoftwareWithGaze/train/data/eth-xgaze-pretrained-model.pth")

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

# Initialize dataset and dataloader
train_dataset = GazeDataset(root_path,train=True)
validation_dataset = GazeDataset(root_path,train=False)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Define loss function
loss_fn = nn.MSELoss()

# Define number of epochs
num_epochs = 50

# Initialize list to store epoch losses and error
train_losses = []
eval_losses = []
mean_angular_errors = []

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}")
    train_loss = 0.0
    eval_loss = 0.0
    angular_error_list = []

    # Iterate over batches
    for imgs, yaws, pitches in tqdm(train_dataloader):
        imgs = imgs.float()
        yaws = yaws.float()
        pitches = pitches.float()
        
        # Forward pass
        yaw_pred, pitch_pred = model(imgs)
        
        # Compute loss
        loss = (loss_fn(yaw_pred, yaws) + loss_fn(pitch_pred, pitches))/2
        train_loss += loss.item() * imgs.size(0)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        for imgs, yaws, pitches in tqdm(eval_dataloader):
            imgs = imgs.float()
            yaws = yaws.float()
            pitches = pitches.float()
            
            # Forward pass
            yaw_pred, pitch_pred = model(imgs)
            
            # Compute loss
            loss = (loss_fn(yaw_pred, yaws) + loss_fn(pitch_pred, pitches))/2
            eval_loss += loss.item() * imgs.size(0)

            # Calculate angular error
            for _yaw_pred, _pitch_pred, _yaw, _pitch in zip(yaw_pred.cpu().numpy(),pitch_pred.cpu().numpy(),yaws.cpu().numpy(),pitches.cpu().numpy()):
                ang_error = angular_error(_yaw_pred, _pitch_pred, _yaw, _pitch)
                angular_error_list.append(ang_error)

    # Update learning rate scheduler
    scheduler.step(train_loss)

    # Print epoch losses
    train_loss /= len(train_dataset)
    eval_loss /= len(validation_dataset)
    print(f"Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")

    mean_angular_error = np.mean(angular_error_list)
    mean_angular_errors.append(mean_angular_error)
    print(f"Mean Angular Error: {mean_angular_error:.4f}")

    train_losses.append(train_loss)
    eval_losses.append(eval_loss)

    # Save the model's weights after each epoch
    checkpoint_path = os.path.join(checkpoints_path, f"epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), checkpoint_path)

    # Plot epoch losses
    fig, ax1 = plt.subplots()
    ax1.plot(train_losses, label="Training Loss", color="tab:blue")
    ax1.plot(eval_losses, label="Evaluation Loss", color="tab:green")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")
    plt.title("Loss Evolution")
    plt.savefig(os.path.join(checkpoints_path, f"epoch_{epoch+1}_loss.png"))
    plt.close()

    # Plot mean angular errors
    fig, ax2 = plt.subplots()
    ax2.plot(mean_angular_errors, label="Mean Angular Error", color="tab:red")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mean Angular Error (degrees)")
    ax2.legend(loc="upper right")
    plt.title("Mean Angular Error Evolution")
    plt.savefig(os.path.join(checkpoints_path, f"epoch_{epoch+1}_angular_error.png"))
    plt.close()
