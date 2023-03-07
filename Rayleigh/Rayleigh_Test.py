# This python script is used to generate rayleigh channel coefficients
import numpy as np
import matplotlib.pyplot as plt
import torch
from math import sqrt

from QPSK_Mod_Demod import QPSK_Modulation

if __name__ == '__main__':
    N = 1024
    # Sinusoidal waveform generation
    # t = np.linspace(0, N, N)
    # x_volts  = 20*np.sin(t/(2*np.pi))
    # x_watts = x_volts ** 2
    # x_db = 10 * np.log10(x_watts)
    qpskmode = 4
    samples=qpskmode*2
    size=(N,qpskmode*2)
    x_int = np.random.randint(0,qpskmode, 10)

    x_volts = 20 * np.sin(x_int / (4 * np.pi))

    # fig, ax1 = plt.subplots()
    # ax1.plot(x_int, x_volts)
    # ax1.grid()
    # ax1.set_xlabel('Time (Number of samples)')
    # ax1.set_ylabel('Cos Wave')
    # plt.title('Generation of Carrier Wave 1')
    # plt.show()

    x_degrees = x_int * 360 / qpskmode + 45  # 45, 135, 225, 315 degrees
    x_radians = x_degrees * np.pi / 180.0  # sin() and cos() takes in radians
    x_symbols = np.cos(x_radians) + 1j * np.sin(x_radians)  # this produces our QPSK complex symbols
    # x_symbols_volts = 20 * np.sin(x_symbols / (2 * np.pi))
    # figure, axis = plt.subplots(3)
    #
    # axis[0].plot(x_volts)
    # axis[0].grid(True)
    # axis[0].set_title('pure')
    # axis[1].plot(x_symbols,x_symbols_volts)
    # axis[1].grid(True)
    # axis[1].set_title('qpsk')
    # axis[2].plot(np.real(x_symbols), np.imag(x_symbols), '.')
    # axis[2].grid(True)
    # axis[2].set_title('qpskconst')
    # plt.show()
    # plt.savefig('BPSK_AWGN_BER_over_SNR.png', format='png')

    # Parameters for simulation
    v = 60 # velocity (meters per second)
    center_freq = 100e6 # RF 100 MHz
    Fs = 2e5 # sample rate 0.2 MHz
    pi = 3.14
    fd = v*center_freq/3e8 # Doppler frequency shift (maximum)
    print("Doppler frequency shift (Max.):", fd)
    t = np.arange(0, 1, 1/Fs) # time vector. (start, stop, step)
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    for i in range(N):
      alpha = (np.random.rand() - 0.5) * 2 * pi
      phi = (np.random.rand() - 0.5) * 2 * pi
      x = x + np.random.randn() * np.cos(2 * pi * fd * t * np.cos(alpha) + phi)
      y = y + np.random.randn() * np.sin(2 * pi * fd * t * np.cos(alpha) + phi)

    z = (1/np.sqrt(N)) * (x + 1j*y) # This is channel response used to convolve with transmitted data or signal
    z_mag = np.abs(z) # Used in plot
    z_mag_dB = 10*np.log10(z_mag) # convert to dB

    # Convolve sinusoidal waveform with Rayleigh Fading channel
    y3 = np.convolve(z, x_symbols)
    y3_mag_dB = 10 * np.log10(np.abs(y3))  # convert to dB
    # y3= z+x_symbols
    # Plots
    figure, axis = plt.subplots(3, 2)
    # axis[0, 0].plot(np.real(x_symbols), np.imag(x_symbols), '.')
    axis[0, 0].plot(x_volts)
    axis[0, 0].grid(True)
    axis[0, 0].set_title("Pure sine wave signal")
    axis[0, 1].plot(np.sin(x_symbols))
    axis[0, 1].grid(True)
    axis[0, 1].set_title("qpsk wave signal(dB)")
    axis[1, 0].plot(z)
    axis[1, 0].set_title("Rayleigh Channel response")
    axis[1, 1].plot(z_mag_dB)
    axis[1, 1].set_title("Rayleigh Channel response (dB)")
    axis[2,0].plot(y3)
    axis[2, 0].set_title("Convolved sine wave signal")
    axis[2, 1].plot(y3_mag_dB)
    axis[2, 1].set_title("Convolved sine wave signal(dB)")

    plt.tight_layout()
    plt.show()
    plt.savefig('BPSK_AWGN_BER_over_SNR.png', format='png')