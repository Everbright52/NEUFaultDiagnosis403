import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import scipy.io as sio


if __name__ == '__main__':
    file_data = [
        ('/home/evb52/DeepLearning/classification/data/bearing_fault_original_data/97.mat', 'X097_DE_time',
         '/home/evb52/DeepLearning/classification/data/image_test_stft/097', 97001),
        ('/home/evb52/DeepLearning/classification/data/bearing_fault_original_data/105.mat', 'X105_DE_time',
         '/home/evb52/DeepLearning/classification/data/image_test_stft/105', 105001),
        ('/home/evb52/DeepLearning/classification/data/bearing_fault_original_data/130.mat', 'X130_DE_time',
         '/home/evb52/DeepLearning/classification/data/image_test_stft/130', 130001),
        ('/home/evb52/DeepLearning/classification/data/bearing_fault_original_data/144.mat', 'X144_DE_time',
         '/home/evb52/DeepLearning/classification/data/image_test_stft/144', 144001),
        ('/home/evb52/DeepLearning/classification/data/bearing_fault_original_data/169.mat', 'X169_DE_time',
         '/home/evb52/DeepLearning/classification/data/image_test_stft/169', 169001),
        ('/home/evb52/DeepLearning/classification/data/bearing_fault_original_data/185.mat', 'X185_DE_time',
         '/home/evb52/DeepLearning/classification/data/image_test_stft/185', 185001),
        ('/home/evb52/DeepLearning/classification/data/bearing_fault_original_data/209.mat', 'X209_DE_time',
         '/home/evb52/DeepLearning/classification/data/image_test_stft/209', 209001),
        ('/home/evb52/DeepLearning/classification/data/bearing_fault_original_data/222.mat', 'X222_DE_time',
         '/home/evb52/DeepLearning/classification/data/image_test_stft/222', 222001),
        ('/home/evb52/DeepLearning/classification/data/bearing_fault_original_data/234.mat', 'X234_DE_time',
         '/home/evb52/DeepLearning/classification/data/image_test_stft/234', 234001),
        ('/home/evb52/DeepLearning/classification/data/bearing_fault_original_data/3005.mat', 'X048_DE_time',
         '/home/evb52/DeepLearning/classification/data/image_test_stft/3005', 3005001),
        # Add more file paths and variable names here
    ]

    # Set the wavelet family and parameters for the CWT

    sampling_rate = 100000

    for file_path, variable_name, save_path, naming_rule in file_data:
        mat_file = sio.loadmat(file_path)
        time_data = mat_file[variable_name].flatten()

        window_size = 1000
        n_images = 500
        N = len(time_data)
        n_windows = n_images - 1
        overlap = (window_size * n_images - N) // n_windows
        # print("变量名：{}  总点数 N = {}  overlap = {}".format(variable_name, N, overlap))


        for i in range(n_images):
            # Calculate the start and end indices for the current window
            start = i * (window_size - overlap)
            end = start + window_size

            # Make sure the end index does not exceed the length of the data
            if end > len(time_data):
                end = len(time_data)

            # # Extract the current window of data
            data = time_data[start:end]
            print("i = {}   起始点:{}   终止点:{}   两者相差：{}  naming_rule = {}".format(i, start, end, len(data),
                                                                                        naming_rule + i))


            # Compute the STFT of the current window
            _, _, Zxx = stft(data, fs=sampling_rate, window='hann', nperseg=window_size, noverlap=overlap)

            # Plot and save the STFT magnitude as an image
            plt.imshow(np.abs(Zxx), extent=[0, 1, 0, sampling_rate / 2], cmap='jet', aspect='auto')
            plt.axis('off')
            plt.savefig(f'{save_path}/{naming_rule + i}.png', dpi=300, bbox_inches='tight', pad_inches=0,
                        transparent=True)
            plt.close()