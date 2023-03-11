# Binary data generator (1's and 0's as per user defined length)
import matplotlib.pyplot as plt
import numpy as np

# Python code to generate binary stream of data
input_data = ""
len1 = 1000 # length of input data
if __name__ == '__main__':

    N = 1000

    x_int =  np.random.randint(0, N, N)  # 0 to 3
    c2 =  # carrier frequency sine wave
    x_degrees = x_int * 360 / 4.0 + 45  # 45, 135, 225, 315 degrees
    x_radians = x_degrees * np.pi / 180.0  # sin() and cos() takes in radians
    x_symbols = np.cos(x_radians) + 1j * np.sin(x_radians)  # this produces our QPSK complex symbols
    plt.plot(np.real(x_symbols), np.imag(x_symbols), '.')
    plt.grid(True)
    plt.show()
    plt.plot(np.cos(x_radians), '.-')
    plt.plot(np.sin(x_radians), '.-')
    plt.legend(['real', 'imag'])
    plt.grid(True)
    plt.show()
    #
    n = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
    plt.plot(np.real(n), '.-')
    plt.plot(np.imag(n), '.-')
    plt.legend(['real', 'imag'])
    plt.show()

    y = x_symbols+n
    plt.plot(np.real(y), '.-')
    plt.plot(np.imag(y), '.-')
    plt.legend(['real', 'imag'])
    plt.show()

