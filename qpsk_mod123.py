from __future__ import division, print_function  # Python 2 compatibility

import math
import os
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np

import commpy.channelcoding.convcode as cc
import commpy.channels as chan
import commpy.links as lk
import commpy.modulation as mod
import commpy.utilities as util
import torch

# Authors: CommPy contributors
# License: BSD 3-Clause

from numpy import arange, sqrt, log10
from numpy.random import seed

from commpy.modulation import QAMModem, kbest, best_first_detector
from scipy import special
from scipy.special import erfc

from get_args import get_args
from plot import get_plots
from utils import get_theo_ber, get_modem, snr_db2sigma, generate_noise_SNR, generate_noise_SNR_Sim

if __name__ == '__main__':
    args = get_args()
    # snrs = np.arange(0, 100, 5)
    # val_loss =(0.9,0.8,0.7)
    # val_ber = (0.6, 0.5, 0.4)
    # val_bler = (1.0,1.0, 1.0)
    # val_loss_fin =[0.0,0.0,0.0]
    # val_ber_fin = [0.0,0.0,0.0]
    # val_bler_fin = [0.0,0.0,0.0]
    # print (snrs.size)
    # for i in range(3):
    #     val_loss_fin[i] =val_loss[i]/snrs.size
    #     val_ber_fin[i] = val_ber[i] / snrs.size
    #     val_bler_fin[i] = val_bler[i] / snrs.size
    #
    # exit()
    # SNR = 19
    # coderate_k = 7
    # coderate_n = 8
    # noise_shape = (100, 10, coderate_n)
    # mod_type ="LDPC"

    # generate_noise_SNR(SNR, noise_shape, args, "AWGN", coderate_k, coderate_n, mod_type)
    # generate_noise_SNR(SNR, noise_shape, args, "Rayleigh", coderate_k, coderate_n, mod_type)
    # generate_noise_SNR(SNR, noise_shape, args, "Rician", coderate_k, coderate_n, mod_type)
    # fwd_noise, encoded_input, input_msg, sim_ber = generate_noise_SNR_Sim(SNR, noise_shape, args, "AWGN", coderate_k,
    #                                                                       coderate_n, mod_type)
    # print("{} AWGN ber: {}".format(mod_type, sim_ber))
    # generate_noise_SNR_Sim(SNR, noise_shape, args, "Rayleigh", coderate_k, coderate_n, mod_type)
    # print("{} Rayleigh ber: {}".format(mod_type, sim_ber))
    # generate_noise_SNR_Sim(SNR, noise_shape, args, "Rician", coderate_k, coderate_n, mod_type)
    # # print("{} Rician ber: {}".format(mod_type, sim_ber))
    # mod_type = "QAM16"
    #
    # generate_noise_SNR(SNR, noise_shape, args, "AWGN", coderate_k, coderate_n, mod_type)
    # generate_noise_SNR(SNR, noise_shape, args, "Rayleigh", coderate_k, coderate_n, mod_type)
    # generate_noise_SNR(SNR, noise_shape, args, "Rician", coderate_k, coderate_n, mod_type)
    # fwd_noise, encoded_input, input_msg, sim_ber,mod = generate_noise_SNR_Sim(SNR, noise_shape, args, "AWGN", coderate_k, coderate_n, mod_type)
    # print("{} AWGN ber: {}".format(mod_type,sim_ber))
    # fwd_noise, encoded_input, input_msg, sim_ber,mod = generate_noise_SNR_Sim(SNR, noise_shape, args, "Rayleigh", coderate_k, coderate_n, mod_type)
    # print("{} Rayleigh ber: {}".format(mod_type,sim_ber))
    # fwd_noise, encoded_input, input_msg, sim_ber,mod = generate_noise_SNR_Sim(SNR, noise_shape, args, "Rician", coderate_k, coderate_n,mod_type)
    # print("{} Rician ber: {}".format(mod_type,sim_ber))
    #
    # exit()
    # coderate_k = 8
    # coderate_n = 9
    # noise_shape = (100, 10, coderate_n)
    #
    # mod_type = "LDPC"
    # generate_noise_SNR(SNR, noise_shape, args, "AWGN", coderate_k, coderate_n, mod_type)
    # mod_type = "POLAR"
    # generate_noise_SNR(SNR, noise_shape, args, "AWGN", coderate_k, coderate_n, mod_type)
    #
    #
    # exit()
    # SNRS =np.arange(-2, 20, 2)
    # idx = 0
    # awgn_bers =np.zeros_like(SNRS, dtype=float)
    # rayleigh_bers = np.zeros_like(SNRS, dtype=float)
    #
    # for SNR in SNRS:
    #     noise_shape = (100,100,3)
    #     input_msg = np.random.choice((0, 1), noise_shape[0]*noise_shape[1]*noise_shape[2] * 4)
    #     msg = input_msg
    #     mod = get_modem('QAM_16')
    #     symbs = mod.modulate(msg)
    #
    #     input_array = np.random.uniform(0, 2, len(symbs))
    #
    #     input_raleigh = symbs + input_array
    #
    #     awgn_bers[idx] =awgn_ber(input_raleigh,msg,SNR,noise_shape,mod)
    #     rayleigh_bers[idx] = rayleigh_ber(input_raleigh,msg,SNR,noise_shape,mod,code_rate=1/3)
    #     idx += 1
    # print('awgn_ber',awgn_bers)
    # print('rayleigh_ber', rayleigh_bers)
    # plt.semilogy(SNRS, awgn_bers, 'o-')
    # # plt.grid()
    # plt.xlabel('Signal to Noise Ration (dB)')
    # plt.ylabel('Bit Error Rate')
    # plt.legend(('awgn snr-ber'))
    # # plt.savefig(test_file_path_exp, format='png', bbox_inches='tight',
    # #             transparent=True, dpi=800)
    # plt.show()
    #
    # plt.semilogy(SNRS, rayleigh_bers, 'o-')
    # # plt.grid()
    # plt.xlabel('Signal to Noise Ration (dB)')
    # plt.ylabel('Bit Error Rate')
    # plt.legend(('rayliegh snr-ber'))
    # # plt.savefig(test_file_path_exp, format='png', bbox_inches='tight',
    # #             transparent=True, dpi=800)
    # plt.show()
    #
    # plt.semilogy(SNRS, awgn_bers, 'o-',SNRS,rayleigh_bers,'o-',SNRS,QAM().calcTheoreticalSER)
    # # plt.grid()
    # plt.xlabel('Signal to Noise Ration (dB)')
    # plt.ylabel('Bit Error Rate')
    # plt.legend(('awgn snr-ber','rayliegh snr-ber'))
    # # plt.savefig(test_file_path_exp, format='png', bbox_inches='tight',
    # #             transparent=True, dpi=800)
    # plt.show()
    # exit()

    # a = np.arange(18).reshape(3,3,2)
    # print('all',a)
    # print(':,:,1',a[:,:,1:])
    # print(':, :, 0',a[:, :, 0:1])
    # exit()
    path = r'C:\WorkSpace\FadingChannels\Swetha_M20AIE317_MTP\Fading\20230513_211438\data_faded'
    plot_path = r'C:\WorkSpace\FadingChannels\Swetha_M20AIE317_MTP\Fading\20230513_211438\plot_faded'
    filename = os.path.join(path,r'attention_data_awgn_lr_0.01_D1All_20230513_211438.txt')

    get_plots(plot_path,filename)
    # test_data = np.loadtxt(os.path.join(path,r'bl_20__k_2_n_3',r'attention_data_test_awgn_lr_0.01_D1bl_20__k_2_n_3_500_20230325_180459.txt')).T
    # plt.semilogy(test_data[3,:],test_data[4,:],'--o')
    # plt.show()
    exit()
#     test_data = np.loadtxt(test_filename, usecols=[0, 1, 2]).T
#     snrs = test_data[0, :]
#     ber = test_data[1, :]
#     bler = test_data[2, :]
#     mean_BER_theoretical = np.zeros(len(snrs))
#     ber_db  = np.zeros(len(snrs))
#     test_file_path = os.path.join(path, 'rayleigh_test_both' + str(timestamp) + '.png')
#     test_file_path_act = os.path.join(path, 'rayleigh_test_act_' + str(timestamp) + '.png')
#     test_file_path_exp = os.path.join(path, 'rayleigh_test_exp_' + str(timestamp) + '.png')
#     ber_snr_noise = (0.39027778 ,0.32083333 ,0.3375,     0.28333333, 0.28680556, 0.20138889
# , 0.18055556, 0.1337963,  0.09305556, 0.04305556, 0.02465278)
#
#     i = 0
#     for SNR in snrs:
#         # calculate theoretical BER for BPSK
#         mean_BER_theoretical[i] = get_theo_ber(SNR)
#         ber_db[i] = 10 * math.log10(ber[i])
#         i += 1
#
#     plt.semilogy(snrs, ber, 'o-', snrs, ber_snr_noise, 'o-')
#     plt.grid()
#     plt.xlabel('Signal to Noise Ration (dB)')
#     plt.ylabel('Bit Error Rate')
#     plt.legend(('test snr-ber', 'theo snr-ber'))
#     plt.savefig(test_file_path, format='png', bbox_inches='tight',
#                 transparent=True, dpi=800)
#     plt.show()
#
#     plt.semilogy(snrs, ber, 'o-')
#     plt.grid()
#     plt.xlabel('Signal to Noise Ration (dB)')
#     plt.ylabel('Bit Error Rate')
#     plt.legend('test snr-ber')
#     plt.savefig(test_file_path_act, format='png', bbox_inches='tight',
#                 transparent=True, dpi=800)
#     plt.show()
#
#     plt.semilogy(snrs, mean_BER_theoretical, 'o-')
#     plt.grid()
#     plt.xlabel('Signal to Noise Ration (dB)')
#     plt.ylabel('Bit Error Rate')
#     plt.legend('theo snr-ber')
#     plt.savefig(test_file_path_exp, format='png', bbox_inches='tight',
#                 transparent=True, dpi=800)
#     plt.show()
#     exit()
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

    # Modem : QAM
    # modem = mod.QAMModem(4)
    QAM16 =  mod.QAMModem(16)
    RayleighChannel =  chan.MIMOFlatChannel(3,3)
    RayleighChannel.uncorr_rayleigh_fading(complex)

    # AWGN channel
    # channels = chan.SISOFlatChannel(None, (1 + 0j,0j) )
    # channels = chan.SISOFlatChannel(None, (0j,1 ))
    # channels = chan.MIMOFlatChannel(4, 4)
    # channels.uncorr_rayleigh_fading(complex)
    # SNR range to test
    SNRs = (-2, 0, 2, 4 , 6, 8, 10, 12, 14, 16,18)
    # snr_interval = ((4) - (-1.5)) * 1.0 / (12- 1)
    # SNRs = [snr_interval * item + (-1.5) for item in range(12)]


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
    # desired[n, m] = exp(1j * 2 * pi * (n * 1 * cos(0.5) - m * 0.1 * cos(2)))
    print(BERs)
    mean_BER_theoretical = np.zeros(len(SNRs))
    desired = np.zeros(len(SNRs))
    i=0
    for SNR in SNRs:
        # calculate theoretical BER for BPSK
        # SNR_lin = 10 ** (SNR / 10)
        # mean_BER_theoretical[i] = 1 / 2 * special.erfc(np.sqrt(SNR_lin))
        # desired[i] = erfc(sqrt(10 ** (SNR/ 10) / 2)) / 2
        mean_BER_theoretical[i] = get_theo_ber(SNR)
        i += 1
    print('desiredber :',desiredber)
    print('mean_BER_theoretical ',mean_BER_theoretical)
    plt.semilogy(SNRs, BERs)
    plt.grid()
    plt.xlabel('Signal to Noise Ration (dB)')
    plt.ylabel('Bit Error Rate')
    plt.legend(('Hard demodulation'))
    plt.show()

    # plt.semilogy(SNRs, BERs, 'o-', SNRs, desired, 'o-')
    # plt.grid()
    # plt.xlabel('Signal to Noise Ration (dB)')
    # plt.ylabel('Bit Error Rate')
    # plt.legend(('Hard demodulation', 'desired'))
    # plt.show()

    plt.semilogy(SNRs, BERs, 'o-', SNRs, mean_BER_theoretical, 'o-')
    plt.grid()
    plt.xlabel('Signal to Noise Ration (dB)')
    plt.ylabel('Bit Error Rate')
    plt.legend(('Hard demodulation', 'math'))
    plt.show()

