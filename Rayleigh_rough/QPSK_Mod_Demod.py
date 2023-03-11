import matplotlib.pyplot as plt
import numpy as np


def QPSK_Modulation(T=1, nb=100, SNR=0,qpsk_mode_type=4, plot_file=''):
    # T - Baseband signal width, which is frequency
    # nb - Define the number of bits transmitted
    delta_T = T / (nb * (qpsk_mode_type / 4))  # sampling interval
    fs = 1 / delta_T  # Sampling frequency
    fc = 10 / T  # Carrier frequency
    # SNR - Signal to noise ratio
    t = np.arange(0, nb * T, delta_T)
    N = len(t)

    # Generate baseband signal
    data = [1 if x > 0.5 else 0 for x in np.random.randn(1, nb)[0]]
    # Call the random function to generate any 1*nb matrix from 0 to 1, which is greater than 0.5 as 1 and less than 0.5 as 0
    data0 = []  # Create a 1*nb/delta_T zero matrix
    for q in range(nb):
        data0 += [data[q]] * int(1 / delta_T)  # Convert the baseband signal into the corresponding waveform signal

    # Modulation signal generation
    data1 = []  # Create a 1*nb/delta_T zero matrix
    datanrz = np.array(data) * 2 - 1  # Convert the baseband signal into a polar code, mapping
    for q in range(nb):
        data1 += [datanrz[q]] * int(1 / delta_T)  # Change the polarity code into the corresponding waveform signal

    idata = datanrz[0:(
                nb - 1):2]  # Serial and parallel conversion, separate the odd and even bits, the interval is 2, i is the odd bit q is the even bit
    qdata = datanrz[1:nb:2]
    ich = []  # Create a 1*nb/delta_T/2 zero matrix to store parity data later
    qch = []
    for i in range(int(nb / 2)):
        ich += [idata[i]] * int(1 / delta_T)  # Odd bit symbols are converted to corresponding waveform signals
        qch += [qdata[i]] * int(1 / delta_T)  # Even bit symbols are converted to corresponding waveform signals

    a = []  # Cosine function carrier
    b = []  # Sine function carrier
    for j in range(int(N / 2)):
        a.append(np.math.sqrt(2 / T) * np.math.cos(2 * np.math.pi * fc * t[j]))  # Cosine function carrier
        b.append(np.math.sqrt(2 / T) * np.math.sin(2 * np.math.pi * fc * t[j]))  # Sine function carrier
    # ibase = np.array(data0[0:(nb - 1):2]) * np.array(a)
    idata1 = np.array(ich) * np.array(a)  # Odd-digit data is multiplied by the cosine function to get a modulated signal
    qdata1 = np.array(qch) * np.array(b)  # Even-digit data is multiplied by the cosine function to get another modulated signal
    s = idata1 + qdata1  # Combine the odd and even data, s is the QPSK modulation signal
    if plot_file != '':
        sample_size = N / (qpsk_mode_type/2)
        plt.figure(figsize=(20, 12))
        plt.subplot(4, 1, 1)
        plt.plot(np.sin(t))
        plt.title('Pure', fontsize=10)
        plt.subplot(4, 1, 2)
        plt.plot(idata1)
        plt.title('In-phase branch I', fontsize=10)
        plt.axis([0, sample_size, -3, 3])
        plt.subplot(4, 1, 3)
        plt.plot(qdata1)
        plt.title('Orthogonal branch Q', fontsize=10)
        plt.axis([0, sample_size, -3, 3])
        plt.subplot(4, 1, 4)
        plt.plot(s)
        plt.title('Modulated signal', fontsize=10)
        plt.axis([0, sample_size, -3, 3])
        plt.savefig(plot_file)
        plt.show()

    return N, t, idata1, qdata1, s
