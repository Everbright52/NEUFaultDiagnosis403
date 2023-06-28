import numpy as np
import ssqueezepy as ssq
import matplotlib.pyplot as plt
import scipy.io as sio

def ssq_scale2freq(scales, samprate):
    freq = scales.flatten()
    freq = 1 / freq * (samprate / 2)
    return freq



if __name__ == '__main__':
    mat_file = sio.loadmat('/home/evb52/DeepLearning/classification/data/bearing_fault_original_data/130.mat')
    data = mat_file['X130_DE_time'].flatten()
    samprate = 100000

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
    overlap = 757

    n_windows = len(data) // (window_size - overlap)

    print("n_windows = {}".format(n_windows))
    # for i in range(n_windows):
    #     start = i * (window_size - overlap)
    #     end = start + window_size
    #     if end > len(data):
    #         break
    #
    #     segment = data[start:end]
    #
    #     _, cfs, _, cfs_scales, *_ = ssq.ssq_cwt(segment, wavelet=wavelet, scales=scales, padtype=padtype, fs=samprate)
    #     freq = ssq_scale2freq(cfs_scales, samprate)
    #
    #     plt.axis('off')
    #     plt.imshow(np.abs(cfs), cmap='jet', aspect='auto', origin='lower', extent=[0, 1, freq[-1], freq[0]])
    #     plt.savefig(f'/home/evb52/DeepLearning/classification/data/image/3005/{3005001 + i}.png', dpi=300,
    #                 bbox_inches='tight', pad_inches=0, transparent=True)
    #     plt.close()
