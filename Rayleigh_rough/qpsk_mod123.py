from __future__ import division, print_function  # Python 2 compatibility

import math

import matplotlib.pyplot as plt
import numpy as np

import commpy.channelcoding.convcode as cc
import commpy.channels as chan
import commpy.links as lk
import commpy.modulation as mod
import commpy.utilities as util

# Authors: CommPy contributors
# License: BSD 3-Clause

from numpy import arange, sqrt, log10
from numpy.random import seed

from commpy.modulation import QAMModem, kbest, best_first_detector

if __name__ == '__main__':
    # Authors: CommPy contributors
    # License: BSD 3-Clause

    #
    #
    # # =============================================================================
    # # Convolutional Code 1: G(D) = [1+D^2, 1+D+D^2]
    # # Standard code with rate 1/2
    # # =============================================================================
    #
    # # Number of delay elements in the convolutional encoder
    # memory = np.array(2, ndmin=1)
    #
    # # Generator matrix
    # g_matrix = np.array((0o5, 0o7), ndmin=2)
    #
    # # Create trellis data structure
    # trellis1 = cc.Trellis(memory, g_matrix)
    #
    # # =============================================================================
    # # Convolutional Code 1: G(D) = [1+D^2, 1+D^2+D^3]
    # # Standard code with rate 1/2
    # # =============================================================================
    #
    # # Number of delay elements in the convolutional encoder
    # memory = np.array(3, ndmin=1)
    #
    # # Generator matrix (1+D^2+D^3 <-> 13 or 0o15)
    # g_matrix = np.array((0o5, 0o15), ndmin=2)
    #
    # # Create trellis data structure
    # trellis2 = cc.Trellis(memory, g_matrix)
    #
    # # =============================================================================
    # # Convolutional Code 2: G(D) = [[1, 0, 0], [0, 1, 1+D]]; F(D) = [[D, D], [1+D, 1]]
    # # RSC with rate 2/3
    # # =============================================================================
    #
    # # Number of delay elements in the convolutional encoder
    # memory = np.array((1, 1))
    #
    # # Generator matrix & feedback matrix
    # g_matrix = np.array(((1, 0, 0), (0, 1, 3)))
    # feedback = np.array(((2, 2), (3, 1)))
    #
    # # Create trellis data structure
    # trellis3 = cc.Trellis(memory, g_matrix, feedback, 'rsc')
    #
    # # =============================================================================
    # # Basic example using homemade counting and hard decoding
    # # =============================================================================
    #
    # # Traceback depth of the decoder
    # tb_depth = None  # Default value is 5 times the number or memories
    #
    # for trellis in (trellis1, trellis2, trellis3):
    #     for i in range(10):
    #         # Generate random message bits to be encoded
    #         message_bits = np.random.randint(0, 2, 1000)
    #
    #         # Encode message bits
    #         coded_bits = cc.conv_encode(message_bits, trellis)
    #
    #         # Introduce bit errors (channel)
    #         coded_bits[np.random.randint(0, 1000)] = 0
    #         coded_bits[np.random.randint(0, 1000)] = 0
    #         coded_bits[np.random.randint(0, 1000)] = 1
    #         coded_bits[np.random.randint(0, 1000)] = 1
    #
    #         # Decode the received bits
    #         decoded_bits = cc.viterbi_decode(coded_bits.astype(float), trellis, tb_depth)
    #
    #         num_bit_errors = util.euclid_dist(message_bits, decoded_bits[:len(message_bits)])
    #         # num_bit_errors += np.bitwise_xor(msg, decoded_bits[:len(msg)].astype(int)).sum()
    #         if num_bit_errors != 0:
    #             print(num_bit_errors, "Bit Errors found!")
    #         elif i == 9:
    #             print("No Bit Errors :)")

    # ==================================================================================================
    # Complete example using Commpy features and compare hard and soft demodulation. Example with code 1
    # ==================================================================================================

    # Modem : QPSK
    # modem = mod.QAMModem(4)
    QAM16 =  mod.QAMModem(16)
    RayleighChannel =  chan.MIMOFlatChannel(4, 4)
    RayleighChannel.uncorr_rayleigh_fading(complex)

    # AWGN channel
    # channels = chan.SISOFlatChannel(None, (1 + 0j,0j) )
    # channels = chan.SISOFlatChannel(None, (0j,1 ))
    # channels = chan.MIMOFlatChannel(4, 4)
    # channels.uncorr_rayleigh_fading(complex)
    # SNR range to test
    SNRs = np.arange(0, 21, 5) + 10 * log10(QAM16.num_bits_symbol)


    def receiver(y, h, constellation, noise_var):
        return QAM16.demodulate(kbest(y, h, constellation, 16), 'hard')

    # Build model from parameters
    # code_rate = trellis3.k / trellis3.n
    code_rate = 1/3

    model = lk.LinkModel(QAM16.modulate, RayleighChannel, receiver,
                            QAM16.num_bits_symbol, QAM16.constellation, QAM16.Es)
    # Test
    BERs = lk.link_performance(model, SNRs, 5e5, 200, 720, model.rate)
    desiredber = (2e-1, 1e-1, 3e-2, 2e-3, 4e-5)
    plt.semilogy(SNRs, BERs, 'o-',SNRs, desiredber, 'o-')
    plt.grid()
    plt.xlabel('Signal to Noise Ration (dB)')
    plt.ylabel('Bit Error Rate')
    plt.legend(('Hard demodulation','desired'))
    plt.show()

