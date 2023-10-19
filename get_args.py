import argparse
import os
import time

def get_args():
    ################################
    # Setup Parameters and get args
    ################################
    parser = argparse.ArgumentParser()
    parser.add_argument('-encoder', choices=['Turboae_rate3_rnn',  # TurboAE Encoder, rate 1/3, RNN
                                             'TurboAE_rate3_rnn_sys',
                                             # TurboAE Encoder, rate 1/3, Systematic Bit hard coded.
                                             'TurboAE_rate3_cnn',
                                             # TurboAE Encoder, rate 1/3, Same Shape 1D CNN encoded.
                                             'TurboAE_rate3_cnn_dense',
                                             # Dense Encoder of TurboAE Encoder, rate 1/3, Same Shape 1D CNN encoded.
                                             'TurboAE_rate3_cnn2d',
                                             # TurboAE Encoder, rate 1/3, Same Shape 2D CNN encoded.
                                             'TurboAE_rate3_cnn2d_dense',
                                             'TurboAE_rate2_rnn',  # TurboAE Encoder, rate 1/2, RNN
                                             'TurboAE_rate2_cnn',
                                             # TurboAE Encoder, rate 1/2, Same Shape 1D CNN encoded.(TBD)
                                             'rate3_cnn',  # CNN Encoder, rate 1/3. No Interleaver
                                             'rate3_cnn2d',
                                             'Turbo_rate3_757',  # Turbo Code, rate 1/3, 757.
                                             'Turbo_rate3_lte',  # Turbo Code, rate 1/3, LTE.
                                             'turboae_2int',  # experimental, use multiple interleavers
                                             ],
                        default='TurboAE_rate3_cnn')

    parser.add_argument('-decoder', choices=['TurboAE_rate3_rnn',  # TurboAE Decoder, rate 1/3
                                             'TurboAE_rate3_cnn',
                                             # TurboAE Decoder, rate 1/3, Same Shape 1D CNN Decoder
                                             'TurboAE_rate3_cnn_dense',  # Dense Encoder of TurboAE Decoder, rate 1/3
                                             'TurboAE_rate3_cnn_2inter',  # TurboAE rate 1/3 CNN with 2 interleavers!
                                             'TurboAE_rate3_cnn2d',
                                             # TurboAE Encoder, rate 1/3, Same Shape 2D CNN encoded.
                                             'TurboAE_rate3_cnn2d_dense',
                                             'TurboAE_rate2_rnn',  # TurboAE Decoder, rate 1/2
                                             'TurboAE_rate2_cnn',  # TurboAE Decoder, rate 1/2
                                             'nbcjr_rate3',  # NeuralBCJR Decoder, rate 1/3, allow ft size.
                                             'rate3_cnn',  # CNN Encoder, rate 1/3. No Interleaver
                                             'rate3_cnn2d',
                                             'turboae_2int',  # experimental, use multiple interleavers
                                             ],
                        default='TurboAE_rate3_cnn')
    #################################
    # Experimetal
    #####################################
    parser.add_argument('--is_k_same_code', action='store_true', default=False,
                        help='train with same code for multiple times')
    ################################################################
    # Channel related parameters
    ################################################################

    # Channel parameters
    parser.add_argument('-vv',type=float, default=5, help ='only for t distribution channel')

    parser.add_argument('-radar_prob',type=float, default=0.05, help ='only for radar distribution channel')
    parser.add_argument('-radar_power',type=float, default=5.0, help ='only for radar distribution channel')

    # continuous channels training algorithms
    parser.add_argument('-train_enc_channel_low', type=float, default  = 0.0)
    parser.add_argument('-train_enc_channel_high', type=float, default = 100)
    parser.add_argument('-train_dec_channel_low', type=float, default  = 0.0)
    parser.add_argument('-train_dec_channel_high', type=float, default = 100.0)

    parser.add_argument('-init_nw_weight', type=str, default='default')

    ################################################################
    # TurboAE encoder/decoder parameters
    ################################################################
    parser.add_argument('-joint_train', type=int, default=0, help='if 1, joint train enc+dec, 0: seperate train')
    parser.add_argument('-enc_rnn', choices=['gru', 'lstm', 'rnn'], default='gru')
    parser.add_argument('-dec_rnn', choices=['gru', 'lstm', 'rnn'], default='gru')

    # CNN/RNN related
    parser.add_argument('-enc_num_layer', type=int, default=3)
    parser.add_argument('-dec_num_layer', type=int, default=3)


    parser.add_argument('-dec_num_unit', type=int, default=100, help = 'This is CNN number of filters, and RNN units')
    parser.add_argument('-enc_num_unit', type=int, default=25, help = 'This is CNN number of filters, and RNN units')

    parser.add_argument('-enc_act', choices=['tanh', 'selu', 'relu', 'elu', 'sigmoid', 'linear'], default='elu', help='only elu works')
    parser.add_argument('-dec_act', choices=['tanh', 'selu', 'relu', 'elu', 'sigmoid', 'linear'], default='linear')

    ################################################################
    # Training ALgorithm related parameters
    ################################################################
    parser.add_argument('-num_train_dec', type=int, default=5, help ='')
    parser.add_argument('-num_train_enc', type=int, default=1, help ='')
    parser.add_argument('-num_train_mod', type=int, default=1, help='')
    parser.add_argument('-num_train_demod', type=int, default=5, help='')
    parser.add_argument('-num_iteration', type=int, default=6)
    # CNN related
    parser.add_argument('-enc_kernel_size', type=int, default=5)
    parser.add_argument('-dec_kernel_size', type=int, default=5)
    parser.add_argument('-extrinsic', type=int, default=1)
    parser.add_argument('-num_iter_ft', type=int, default=5)


    parser.add_argument('-mod_pc',
                        choices=['qpsk', 'symbol_power', 'block_power'],
                        default='qpsk')

    parser.add_argument('-is_interleave', type=int, default=1,
                        help='0 is not interleaving, 1 is fixed interleaver, >1 is random interleaver')
    parser.add_argument('-is_same_interleaver', type=int, default=1,
                        help='not random interleaver, potentially finetune?')

    parser.add_argument('-mod_num_layer', type=int, default=1, help='')
    parser.add_argument('-mod_num_unit', type=int, default=20, help='')
    parser.add_argument('-demod_num_layer', type=int, default=1, help='')
    parser.add_argument('-demod_num_unit', type=int, default=20, help='')
    parser.add_argument('-mod_lr', type=float, default=0.005, help='modulation leanring rate')
    parser.add_argument('-demod_lr', type=float, default=0.005, help='demodulation leanring rate')

    parser.add_argument('-dropout',type=float, default=0.0)

    parser.add_argument('-snr_test_start', type=float, default=-0)
    parser.add_argument('-snr_test_end', type=float, default=100)
    parser.add_argument('-snr_points', type=int, default=20)

    parser.add_argument('-batch_size', type=int, default=5)
    parser.add_argument('-num_epoch', type=int, default=2)
    parser.add_argument('-test_ratio', type=int, default=1,help = 'only for high SNR testing')
    # block length related
    # parser.add_argument('-block_len',type = tuple , default=(10,20,50,100))
    parser.add_argument('-block_len', type=tuple, default=(2, 4))
    # code rate is k/n, so that enable multiple code rates. This has to match the encoder/decoder nw structure.
    # parser.add_argument('-code_rate_k', type = tuple , default=(3,5,7))
    # parser.add_argument('-code_rate_n', type = tuple , default=(4,6,8))
    # parser.add_argument('-code_rate_k', type=tuple, default=(1,1))#, 5, 7))
    # parser.add_argument('-code_rate_n', type=tuple, default=(3,3))#, 6, 8))
    # parser.add_argument('-code_rate_n', type=tuple, default=(3,3))#, 6, 8))
    parser.add_argument('-code_rate_k', type=int, default=1)#, 5, 7))
    parser.add_argument('-code_rate_n', type=int, default=3)#, 6, 8))
    # Non-AWGN, Radar, with -radar_prob, radar_power, associated
    parser.add_argument('-channel',type=tuple, default=('awgn','fading'))

    parser.add_argument('-modtype', type = tuple , default=('QAM2','QAM4')) #'LDPC',
    parser.add_argument('-mod_rate', type=tuple, default=(2,4), help='code: (B, L, R), mode_output (B, L*R/mod_rate, 2)')

    parser.add_argument('-block_len_low', type=int, default=10)
    parser.add_argument('-block_len_high', type=int, default=200)
    parser.add_argument('--is_variable_block_len', action='store_true', default=False,
                        help='training with different block length')

    parser.add_argument('-num_block', type=int, default=100)

    parser.add_argument('-test_channel_mode',
                        choices=['block_norm','block_norm_ste'],
                        default='block_norm')
    parser.add_argument('-train_channel_mode',
                        choices=['block_norm','block_norm_ste'],
                        default='block_norm')
    parser.add_argument('-enc_truncate_limit', type=float, default=0, help='0 means no truncation')

    parser.add_argument('--no_code_norm', action='store_true', default=False,
                        help='the output of encoder is not normalized. Modulation do the work')



    ################################################################
    # STE related parameters
    ################################################################
    parser.add_argument('-enc_quantize_level', type=float, default=2, help = 'only valid for block_norm_ste')
    parser.add_argument('-enc_value_limit', type=float, default=1.0, help = 'only valid for block_norm_ste')
    parser.add_argument('-enc_grad_limit', type=float, default=0.01, help = 'only valid for block_norm_ste')
    parser.add_argument('-enc_clipping', choices=['inputs', 'gradient', 'both', 'none'], default='both',
                        help = 'only valid for ste')

    ################################################################
    # Optimizer related parameters
    ################################################################
    parser.add_argument('-optimizer', choices=['adam', 'lookahead', 'sgd'], default='adam', help = '....:)')
    parser.add_argument('-dec_lr', type = float, default=0.01, help='decoder leanring rate')
    parser.add_argument('-enc_lr', type = float, default=0.01, help='encoder leanring rate')

    ################################################################
    # MISC
    ################################################################
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--rec_quantize', action='store_true', default=False,
                        help='binarize received signal, which will degrade performance a lot')

    parser.add_argument('--print_pos_ber', action='store_true', default=False,
                        help='print positional ber when testing BER')
    parser.add_argument('--print_pos_power', action='store_true', default=False,
                        help='print positional power when testing BER')
    parser.add_argument('--print_test_traj', action='store_true', default=False,
                        help='print test trajectory when testing BER')
    parser.add_argument('--precompute_norm_stats', action='store_true', default=False,
                        help='Use pre-computed mean/std statistics')

    parser.add_argument('-D', type=int, default=1, help='delay')

    parser.add_argument('-BASE_PATH', default=r'C:\WorkSpace\FadingChannels\Swetha_M20AIE317_MTP\Fading')
    parser.add_argument('-LOG_PATH', default= os.path.join(r'C:\WorkSpace\FadingChannels\Swetha_M20AIE317_MTP\Fading',time.strftime('%Y%m%d_%H%M%S', time.localtime()),r'logs_faded'))
    parser.add_argument('-DATA_PATH', default= os.path.join(r'C:\WorkSpace\FadingChannels\Swetha_M20AIE317_MTP\Fading',time.strftime('%Y%m%d_%H%M%S', time.localtime()),r'data_faded'))
    parser.add_argument('-MODEL_PATH', default= os.path.join(r'C:\WorkSpace\FadingChannels\Swetha_M20AIE317_MTP\Fading',time.strftime('%Y%m%d_%H%M%S', time.localtime()),r'model_faded'))
    parser.add_argument('-PLOT_PATH', default= os.path.join(r'C:\WorkSpace\FadingChannels\Swetha_M20AIE317_MTP\Fading',time.strftime('%Y%m%d_%H%M%S', time.localtime()),r'plot_faded'))
    args = parser.parse_args()

    return args