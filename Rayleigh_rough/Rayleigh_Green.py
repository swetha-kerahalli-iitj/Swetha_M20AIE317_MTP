# Import NumPy:

import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # Create the QPSK symbol map:
    QPSKMapper = np.zero(4,dtype='complex')
    QPSKMapper[0] =1+1j
    QPSKMapper[1] = 1–1*j
    QPSKMapper = (np.sqrt(2) / 2) * np.array(dtype = 'complex',[1+1j, 1–1j, –1+1j, –1–1j],)

    # Randomly generate some QPSK symbols:

    ns = 16
    mapIndex = np.random.randint(0, len(QPSKMapper), ns*4)
    QPSKSymbols = QPSKMapper[mapIndex]

    # Define the samples per symbol, which is also the pulse shaping filter length:

    SPS = 2

    # Upsample the symbols:

    QPSKSymbolsUpsampled = np.zeros(ns*SPS,dtype=complex)
    QPSKSymbolsUpsampled[::SPS] = QPSKSymbols

    # Define the pulse shaping filter:

    pulseShape = np.ones(SPS)

    # Apply the pulse shaping filter:

    QPSKSignal = np.convolve(QPSKSymbolsUpsampled,pulseShape)

    sample_size=ns * 10
    plt.figure(figsize=(30, 12))
    plt.subplot(5, 1, 1)
    plt.plot(np.sin(QPSKSymbolsUpsampled))
    plt.title('Pure sine wave signal', fontsize=10)
    plt.axis([0, sample_size, -3, 3])
    plt.subplot(5, 1, 2)
    # plt.plot(qpsk_mod)
    # plt.title('QPSK Modulated wave signal', fontsize=10)
    # plt.axis([0, sample_size, -3, 3])
    # plt.subplot(5, 1, 3)
    plt.plot(pulseShape)
    plt.title('pulsed', fontsize=10)
    plt.axis([0, sample_size, -3, 3])
    plt.subplot(5, 1,3)
    plt.plot(QPSKSignal.real)
    plt.title('modulated(real)', fontsize=10)
    plt.axis([0, sample_size, -20, 20])
    plt.subplot(5, 1, 4)
    plt.plot(QPSKSignal.img)
    plt.title('modulated(img)', fontsize=10)
    plt.axis([0, sample_size, -20, 20])
    # plt.subplot(5, 1, 5)
    # plt.plot(y3)
    # plt.title('Convolved sine wave signal', fontsize=10)
    # plt.axis([0, sample_size, -100, 100])
    plt.tight_layout()
    # plt.savefig(rayleigh_file)
    plt.show()