import numpy as np
import ssqueezepy as ssq
import matplotlib.pyplot as plt
import scipy.io as sio


def ssq_scale2freq(scales, samprate):
    freq = scales.flatten()
    freq = 1 / freq * (samprate / 2)
    return freq

if __name__ == '__main__':
    file_data = [
        ('/home/evb52/DeepLearning/classification/data/bearing_fault_original_data/97.mat', 'X097_DE_time',
         '/home/evb52/DeepLearning/classification/data/image_test/097', 97001),
        ('/home/evb52/DeepLearning/classification/data/bearing_fault_original_data/105.mat', 'X105_DE_time',
         '/home/evb52/DeepLearning/classification/data/image_test/105', 105001),
        ('/home/evb52/DeepLearning/classification/data/bearing_fault_original_data/130.mat', 'X130_DE_time',
         '/home/evb52/DeepLearning/classification/data/image_test/130', 130001),
        ('/home/evb52/DeepLearning/classification/data/bearing_fault_original_data/144.mat', 'X144_DE_time',
         '/home/evb52/DeepLearning/classification/data/image_test/144', 144001),
        ('/home/evb52/DeepLearning/classification/data/bearing_fault_original_data/169.mat', 'X169_DE_time',
         '/home/evb52/DeepLearning/classification/data/image_test/169', 169001),
        ('/home/evb52/DeepLearning/classification/data/bearing_fault_original_data/185.mat', 'X185_DE_time',
         '/home/evb52/DeepLearning/classification/data/image_test/185', 185001),
        ('/home/evb52/DeepLearning/classification/data/bearing_fault_original_data/209.mat', 'X209_DE_time',
         '/home/evb52/DeepLearning/classification/data/image_test/209', 209001),
        ('/home/evb52/DeepLearning/classification/data/bearing_fault_original_data/222.mat', 'X222_DE_time',
         '/home/evb52/DeepLearning/classification/data/image_test/222', 222001),
        ('/home/evb52/DeepLearning/classification/data/bearing_fault_original_data/234.mat', 'X234_DE_time',
         '/home/evb52/DeepLearning/classification/data/image_test/234', 234001),
        ('/home/evb52/DeepLearning/classification/data/bearing_fault_original_data/3005.mat', 'X048_DE_time',
         '/home/evb52/DeepLearning/classification/data/image_test/3005', 3005001),
        # Add more file paths and variable names here
    ]
    samprate = 100000
    window_size = 1000
    n_images = 500


    for file_path, variable_name, save_path, naming_rule in file_data:
        mat_file = sio.loadmat(file_path)
        data = mat_file[variable_name].flatten()


        N = len(data)
        M = ssq.utils.p2up(N)[0]
        wavelet = 'gmw'
        padtype = 'reflect'
        scaletype = 'log-piecewise'
        preset = 'minimal'
        nv = 16
        downsample = 1
        show_last = 20

        wavelet = ssq.Wavelet(wavelet, N=M)

        data = data - np.mean(data)
        min_scale = 10
        max_scale = 100
        scales = ssq.utils.make_scales(M, min_scale, max_scale, nv, scaletype, wavelet, downsample)

        window_size = 1000
        n_images = 500

        n_windows = n_images - 1
        overlap = (window_size * n_images - N) // n_windows
        # print("变量名：{}  总点数 N = {}  overlap = {}".format(variable_name, N, overlap))

        for i in range(n_images):
            start = i * (window_size - overlap)
            end = start + window_size

            if end > len(data):
                end = len(data)

            segment = data[start:end]
            if(i == 499):
                print("i = {}   起始点:{}   终止点:{}   两者相差：{}  naming_rule = {}".format(i, start, end, len(segment),
                                                                                          naming_rule + i))


            # _, cfs, _, cfs_scales, *_ = ssq.ssq_cwt(segment, wavelet=wavelet, scales=scales, padtype=padtype, fs=samprate)
            # freq = ssq_scale2freq(cfs_scales, samprate)
            #
            # plt.axis('off')
            # plt.imshow(np.abs(cfs), cmap='jet', aspect='auto', origin='lower', extent=[0, 1, freq[-1], freq[0]])
            # plt.savefig(f'{save_path}/{naming_rule + i}.png', dpi=300,
            #             bbox_inches='tight', pad_inches=0, transparent=True)
            # plt.close()
