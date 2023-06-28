import numpy as np
import matplotlib.pyplot as plt

# Load time results
time_results = np.load('/home/evb52/DeepLearning/classification/model_and_metric/time_results_100_0.0001.npz', allow_pickle=True)
model_names = time_results.files

# Initialize lists to store time values
load_data_times = []
load_weight_times = []
total_times = []
avg_times = []

# Extract time values for each model
for model_name in model_names:
    model_time = time_results[model_name].item()
    load_data_times.append(model_time['load_data_time'])
    load_weight_times.append(model_time['load_weight_time'])
    total_times.append(model_time['total_time'])
    avg_times.append(model_time['avg_time'])

# Set up subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Model Time Comparison')

# Plot load data time comparison
axs[0, 0].bar(model_names, load_data_times)
axs[0, 0].set_title('Load Data Time')
axs[0, 0].set_ylabel('Time (s)')

# Plot load weight time comparison
axs[0, 1].bar(model_names, load_weight_times)
axs[0, 1].set_title('Load Weight Time')
axs[0, 1].set_ylabel('Time (s)')

# Plot total time comparison
axs[1, 0].bar(model_names, total_times)
axs[1, 0].set_title('Total Time')
axs[1, 0].set_ylabel('Time (ms)')

# Plot average time per prediction comparison
axs[1, 1].bar(model_names, avg_times)
axs[1, 1].set_title('Average Time per Prediction')
axs[1, 1].set_ylabel('Time (ms)')

# Adjust spacing between subplots
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.4)

plt.show()
