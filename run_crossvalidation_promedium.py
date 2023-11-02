import os
import matplotlib.pyplot as plt

# specify the parent directory where the files are located
parent_dir = '/workspaces/AssistanceSoftwareWithGaze/train/output_evaluation'

# initialize a dictionary to store the Mean Angular Error values for each epoch
sum_epoch_mae = {}

info_file = {}

# loop over all subdirectories in the parent directory
for subdir in os.listdir(parent_dir):
    # get the path to the subdirectory
    subdir_path = os.path.join(parent_dir, subdir)

    # skip any files in the parent directory (only process subdirectories)
    if not os.path.isdir(subdir_path):
        continue

    # loop over all files in the subdirectory
    for filename in os.listdir(subdir_path):
        # skip any directories within the subdirectory (only process files)
        if os.path.isdir(os.path.join(subdir_path, filename)):
            continue

        if not filename.endswith('.log'):
            continue

        # read the Mean Angular Error values from the file
        epoch_info = {}
        with open(os.path.join(subdir_path, filename)) as f:
            for line in f:
                if line.startswith('Epoch'):
                    epoch, mae = line.split(':')
                    epoch = epoch.split()[1]
                    mae = float(mae.split('=')[1].strip())

                    epoch_info[epoch] = mae
                    info_file[os.path.basename(subdir_path)] = epoch_info

                    # add the Mean Angular Error value to the dictionary
                    if epoch not in sum_epoch_mae:
                        sum_epoch_mae[epoch] = mae
                    else:
                        sum_epoch_mae[epoch] += mae


# calculate the average and minimum MAE for each epoch
epoch_avg_mae = {}
for epoch in sum_epoch_mae:
    epoch_avg_mae[epoch] = sum_epoch_mae[epoch] / len(os.listdir(parent_dir))

minimum_mae_epoch = min_value = min(epoch_avg_mae, key=lambda k: epoch_avg_mae[k])
minimum_mae = epoch_avg_mae[minimum_mae_epoch]

graph_bar_info = {}
for key in info_file.keys():
    graph_bar_info[key] = info_file[key][minimum_mae_epoch]

# plot a bar graph of the average epoch MAE values
plt.bar(graph_bar_info.keys(), graph_bar_info.values())
plt.xlabel(f'Models at Epoch = {minimum_mae_epoch}')
plt.ylabel('Mean Angular Error')
plt.title('Average Mean Angular Error per Epoch')

# plot a line for the average minimum MAE value
plt.axhline(minimum_mae, color='red', linestyle='--', label=f'Avg. Min. MAE = {minimum_mae:.3f} at Epoch = {minimum_mae_epoch}')
plt.legend()

plt.show()
