import torch
import torch.optim as optim
import sys, os
import time
from get_args import get_args
import math

import matplotlib.pyplot as plt
import numpy as np

import commpy.channelcoding.convcode as cc
import commpy.channels as chan
# import commpy.links as lk
from moddemod_qpsk_links import LinkModel
import commpy.modulation as mod
import commpy.utilities as util
from scipy import special
from channel_enc_dec import Channel_AE
from moddemod_train import train,validate,test

from encoder import ENC
from decoder import DEC


attnFilename=[]
# utils for logger
class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


if __name__ == '__main__':
    #################################################
    # load args & setup logger
    #################################################
    start_time = time.time()

    args = get_args()
    BASE_PATH =args.BASE_PATH
    LOG_PATH =args.LOG_PATH
    DATA_PATH =args.DATA_PATH
    MODEL_PATH =args.MODEL_PATH
    PLOT_PATH = args.PLOT_PATH
    for path in (LOG_PATH, DATA_PATH, MODEL_PATH,PLOT_PATH):
        if not os.path.isdir(path):
            os.makedirs(path)
    # put all printed things to log file
    if args.init_nw_weight == 'default':
        start_epoch = 1
        timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    else:
        start_epoch = int(args.init_nw_weight.split('_')[2]) + 1
        timestamp = args.init_nw_weight.split('_')[8].split('.')[0]
    formatfile = 'attention_log_' + str(args.channel) + '_lr_' + str(args.enc_lr) + '_D' + str(args.D) + '_' + str(
        args.num_block) + '_' + timestamp + '.txt'
    logfilename = os.path.join(LOG_PATH, formatfile)
    logfile = open(logfilename, 'a')
    sys.stdout = Logger(logfilename, sys.stdout)

    print(args)

    filename = os.path.join(DATA_PATH, 'attention_data_' + str(args.channel) + '_lr_' + str(args.enc_lr) + '_D' + str(
        args.D) + '_' + str(args.num_block) + '_' + timestamp + '.txt')
    filename_test = os.path.join(DATA_PATH, 'attention_data_test_' + str(args.channel) + '_lr_' + str(args.enc_lr) + '_D' + str(
        args.D) + '_' + str(args.num_block) + '_' + timestamp + '.txt')
    attnFilename.append(filename)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("use_cuda: ", use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    #################################################
    # Setup Channel AE: Encoder, Decoder, Channel
    #################################################
    # Modem : QPSK
    QAM16 = mod.QAMModem(16)
    channels = chan.MIMOFlatChannel(4, 4)
    channels.uncorr_rayleigh_fading(complex)

    # SNR range to test
    SNRs = np.arange(0, 21, 5) + 10 * math.log10(QAM16.num_bits_symbol)
    mean_BER_theoretical = np.zeros(len(SNRs))
    i = 0
    for SNR in SNRs:
        # calculate theoretical BER for BPSK
        SNR_lin = 10 ** (SNR / 10)
        mean_BER_theoretical[i] = 1 / 2 * special.erfc(np.sqrt(SNR_lin))
        i += 1

    # Build model from parameters
    code_rate = 1 / 3

    model = Channel_AE(args,channels,QAM16, code_rate).to(device)

    # weight loading
    if args.init_nw_weight == 'default':
        pass

    else:
        pretrained_model = torch.load(args.init_nw_weight)

        try:
            model.load_state_dict(pretrained_model.state_dict(), strict=False)

        except:
            model.load_state_dict(pretrained_model, strict=False)

        model.args = args

    print(model)


    #################################################
    # Training Processes
    #################################################
    report_loss, report_ber, report_bler = [], [], []
    for epoch in range(start_epoch, args.num_epoch + 1):
        epoch_start_time = time.time()
        SNRs = np.arange(0, 21, 5) + 10 * math.log10(QAM16.num_bits_symbol)
        this_loss,this_ber= train(epoch, model, SNRs, code_rate,args)
        this_loss, this_ber= validate(model, SNRs,code_rate, args, use_cuda=use_cuda)
        this_bler = this_loss
        report_loss.append(this_loss)
        report_ber.append(this_ber[0])
        report_bler.append(this_bler)

        data_file = open(filename, 'a')
        data_file.write(str(epoch) + ' ' + str(this_loss) + ' ' + str(this_ber[0]) + ' ' + str(this_bler) + "\n")
        data_file.close()

        # save model per epoch
        modelpath = os.path.join(MODEL_PATH, 'attention_model_' + str(epoch) + '_' + str(args.channel) + '_lr_' + str(
            args.enc_lr) + '_D' + str(args.D) + '_' + str(args.num_block) + '_' + timestamp + '.pt')
        torch.save(model.state_dict(), modelpath)
        print('saved model', modelpath)
        print("each epoch training time: {}s".format(time.time() - epoch_start_time))

    if args.print_test_traj == True:
        print('test loss trajectory', report_loss)
        print('test ber trajectory', report_ber)
        print('test bler trajectory', report_bler)
        print('total epoch', args.num_epoch)

    #################################################
    # Testing Processes
    #################################################

    modelpath = os.path.join(MODEL_PATH,
                             'attention_model_' + str(args.channel) + '_lr_' + str(args.enc_lr) + '_D' + str(
                                 args.D) + '_' + str(args.num_block) + '.pt')

    torch.save(model.state_dict(), modelpath)
    print('saved model', modelpath)


    if args.is_variable_block_len:
        print('testing block length', args.block_len_low)
        test(model, args, block_len=args.block_len_low, use_cuda=use_cuda)
        print('testing block length', args.block_len)
        test(model, args, block_len=args.block_len, use_cuda=use_cuda)
        print('testing block length', args.block_len_high)
        test(model, args, block_len=args.block_len_high, use_cuda=use_cuda)

    else:
        test(model,SNRs, code_rate, filename_test,args, use_cuda=use_cuda)

    data = np.loadtxt(filename, usecols=[0, 2]).T
    test_data = np.loadtxt(filename_test, usecols=[0, 1]).T
    snrs = test_data[0, :]
    ber = test_data[1, :]
    epoch = data[0, :]
    ber_epoch = data[1, :]
    mean_BER_theoretical = np.zeros(len(snrs))
    ber_db= np.zeros(len(snrs))
    i = 0
    for SNR in snrs:
        # calculate theoretical BER for BPSK
        SNR_lin = 10 ** (SNR / 10)
        mean_BER_theoretical[i] = 1 / 2 * special.erfc(np.sqrt(SNR_lin))
        i += 1

    plt.semilogy(epoch, ber_epoch, 'o-')
    plt.grid()
    plt.xlabel('Epoch)')
    plt.ylabel('Bit Error Rate')
    plt.legend('epoch-ber')
    plt.savefig('./attention_D_epcoh_' + str(1) + '.png', format='png', bbox_inches='tight',
                transparent=True, dpi=800)
    plt.show()
    desiredber = (2e-1, 1e-1, 3e-2, 2e-3, 4e-5)
    plt.semilogy(snrs, ber, 'o-',snrs, desiredber, 'o-')
    plt.grid()
    plt.xlabel('Signal to Noise Ration (dB)')
    plt.ylabel('Bit Error Rate')
    plt.legend(('SNR-BER','Theo SNR-BER'))
    plt.savefig('./attention_D_snr_' + str(1) + '.png', format='png', bbox_inches='tight',
                transparent=True, dpi=800)
    plt.show()

    # plt.semilogy( snrs, mean_BER_theoretical, 'o-')
    # plt.grid()
    # plt.xlabel('Signal to Noise Ration (dB)')
    # plt.ylabel('Bit Error Rate')
    # plt.legend('Theo SNR-ber')
    # plt.savefig('./attention_D_snr_theo_' + str(1) + '.png', format='png', bbox_inches='tight',
    #             transparent=True, dpi=800)
    # plt.show()


