#生成同步压缩小波时频图
import numpy as np
import ssqueezepy as ssq
import matplotlib.pyplot as plt
import scipy.io as sio

def ssq_scale2freq(scales, samprate):
    freq = scales.flatten()
    freq = 1 / freq * (samprate / 2)
    return freq

if __name__ == '__main__':
    mat_file = sio.loadmat('/home/evb52/DeepLearning/classification/data/bearing_fault_original_data/97.mat')
    data = mat_file['X097_DE_time'][:1000].flatten()  # 只取前一千个点并展平一维的时域信号
    # data = mat_file['X097_DE_time'].flatten()  # 读取并展平一维的时域信号
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
    min = 10
    max = 100
    # min, max = ssq.utils.cwt_scalebounds(wavelet, N=M, preset=preset)
    scales = ssq.utils.make_scales(M, min, max, nv, scaletype, wavelet, downsample)

    _, cfs, _, cfs_scales, *_ = ssq.ssq_cwt(data, wavelet=wavelet, scales=scales, padtype=padtype, fs=samprate)

    freq = ssq_scale2freq(cfs_scales, samprate)

    plt.axis('off')
    plt.imshow(np.abs(cfs), cmap='jet', aspect='auto', origin='lower')
    plt.show()

