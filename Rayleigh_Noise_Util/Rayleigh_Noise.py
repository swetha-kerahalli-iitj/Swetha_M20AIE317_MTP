# This python script is used to generate rayleigh channel coefficients
import os
import numpy as np
import matplotlib.pyplot as plt
import torch


def Rayleigh_Noise(noise_shape,base_path='', show_plot=True):
    # Parameters for simulation

    v = 60  # velocity (meters per second)
    center_freq = 100e6  # RF 100 MHz
    Fs = 2e5  # sample rate 0.2 MHz
    pi = 3.14
    fd = v * center_freq / 3e8  # Doppler frequency shift (maximum)
    print("Doppler frequency shift (Max.):", fd)
    modulation_file_name=''
    rayleigh_file=''
    if show_plot:
        base_path = r'C:\WorkSpace\Swetha_M20AIE317_MTP\Rayleigh\QPSK'
        modulation_file_name = os.path.join(base_path, r'QPSK_Modulation_q' + str(qpsk_mode_type) + '.png')
        rayleigh_file = os.path.join(base_path, 'Rayleigh_QPSK_q' + str(qpsk_mode_type) + '.png')
    input = torch.randn(noise_shape, dtype=torch.float)
    # input = torch.randint(0, 2, input_shape.shape, dtype=torch.float)
    # (N, pure_x, cosine_x, sine_y, qpsk_mod) = QPSK_Modulation(nb=30, qpsk_mode_type=qpsk_mode_type,
    #plot_file=modulation_file_name)
    N = 1000
    x = np.zeros(input.shape)
    y = np.zeros(input.shape)
    for i in range(N):
        alpha = (np.random.rand() - 0.5) * 2 * pi
        phi = (np.random.rand() - 0.5) * 2 * pi
        x =  np.array(x) + np.array(np.random.randn() * np.cos(2 * pi * fd * input * np.cos(alpha) + phi))
        y = np.array(y)+np.array(np.random.randn() * np.sin(2 * pi * fd * input * np.cos(alpha) + phi))

    z = (1 / np.sqrt(N)) * (x + 1j * y)  # This is channel response used to convolve with transmitted data or signal
    z_mag = np.abs(z)  # Used in plot
    z_mag_dB = 10 * np.log10(z_mag)  # convert to dB

    # # Convolve sinusoidal waveform with Rayleigh Fading channel
    # y3 = np.convolve(z, qpsk_mod)

    if show_plot:
        sample_size = N / (qpsk_mode_type / 2)
        plt.figure(figsize=(30, 12))
        plt.subplot(5, 1, 1)
        plt.plot(np.sin(input))
        plt.title('Pure sine wave signal', fontsize=10)
        plt.axis([0, sample_size, -3, 3])
        plt.subplot(5, 1, 2)
        # plt.plot(qpsk_mod)
        # plt.title('QPSK Modulated wave signal', fontsize=10)
        # plt.axis([0, sample_size, -3, 3])
        # plt.subplot(5, 1, 3)
        plt.plot(z)
        plt.title('Rayleigh Channel response', fontsize=10)
        plt.axis([0, sample_size, -3, 3])
        plt.subplot(5, 1, 4)
        plt.plot(z_mag_dB)
        plt.title('Rayleigh Channel response (dB)', fontsize=10)
        # plt.axis([0, sample_size, -20, 20])
        # plt.subplot(5, 1, 5)
        # plt.plot(y3)
        # plt.title('Convolved sine wave signal', fontsize=10)
        # plt.axis([0, sample_size, -100, 100])
        plt.tight_layout()
        plt.savefig(rayleigh_file)
        plt.show()
    return torch.from_numpy(z)
