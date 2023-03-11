from fractions import Fraction

import torch
import torch.optim as optim
import sys, os
import time
from get_args import get_args
import math

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from inspect import getfullargspec

import commpy.channelcoding.convcode as cc
import commpy.channels as chan
# import commpy.links as lk
from moddemod_qpsk_links import LinkModel
import commpy.modulation as mod
import commpy.utilities as util
from scipy import special

from encoder import ENC
from decoder import DEC


######################################################################################
#
# Trainer, validation, and test for AE code design
#
######################################################################################

def customized_loss(output, X_train, args, size_average=True, noise=None, code=None):
    # output = torch.clamp(output, 0.0, 1.0)
    if size_average == True:
        loss = F.binary_cross_entropy(output, X_train)
    else:
        return [F.binary_cross_entropy(item1, item2) for item1, item2 in zip(output, X_train)]

    return loss


def errors_ber(y_true, y_pred, positions='default'):
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    # myOtherTensor = torch.ne(torch.round(y_true), torch.round(y_pred)).float()
    myOtherTensor = torch.ne(y_true, y_pred).float()
    if positions == 'default':
        res = sum(sum(myOtherTensor)) / (myOtherTensor.shape[0] * myOtherTensor.shape[1])
    else:
        res = torch.mean(myOtherTensor, dim=0).type(torch.FloatTensor)
        for pos in positions:
            res[pos] = 0.0
        res = torch.mean(res)
    return res


def errors_bler(y_true, y_pred, positions='default'):
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    decoded_bits = torch.round(y_pred)
    X_test = torch.round(y_true)
    tp0 = (abs(decoded_bits - X_test)).view([X_test.shape[0], X_test.shape[1]])
    tp0 = tp0.cpu().numpy()

    if positions == 'default':
        bler_err_rate = sum(np.sum(tp0, axis=1) > 0) * 1.0 / (X_test.shape[0])
    else:
        for pos in positions:
            tp0[:, pos] = 0.0
        bler_err_rate = sum(np.sum(tp0, axis=1) > 0) * 1.0 / (X_test.shape[0])

    return bler_err_rate


def train(epoch, model, SNRs, code_rate, args, use_cuda=False, verbose=True):
    device = torch.device("cuda" if use_cuda else "cpu")

    model.train()
    modem = model.modem
    channel_ae = model.channels
    start_time = time.time()
    BERs = np.zeros_like(SNRs, dtype=float)
    # Computations
    for id_SNR in range(len(SNRs)):
        channel_ae.set_SNR_dB(SNRs[id_SNR], float(code_rate), modem.Es)
        bit_send = 0
        bit_err = 0
        rate = Fraction(code_rate).limit_denominator(100)
        divider = (Fraction(1, modem.num_bits_symbol * channel_ae.nb_tx) * 1 / rate).denominator
        send_chunk = args.send_chunk
        send_chunk = max(divider, send_chunk // divider * divider)

        receive_size = channel_ae.nb_tx * modem.num_bits_symbol
        ber=0.0
        while bit_send < args.batch_size and bit_err < args.err_min:
            ber,output,X_train,fwd_noise = model(id_SNR, ber,args.send_chunk, receive_size)
            bit_send += send_chunk
        BERs[id_SNR] = ber / bit_send
        if ber < args.err_min:
            break

    end_time = time.time()
    # train_loss = train_loss / (args.num_block / args.batch_size)
    train_loss = 0
    if verbose:
        print('====> Epoch: {} Average loss: {:.8f}'.format(epoch, train_loss), \
              ' running time', str(end_time - start_time))
        print('====> SNR vs BER ==> SNR {}, ber {}'.format(SNRs, BERs), \
              ' running time', str(end_time - start_time))

    return train_loss, BERs


def validate(model,  SNRs, code_rate, args, use_cuda=False, verbose=True):
    device = torch.device("cuda" if use_cuda else "cpu")

    model.eval()
    modem = model.modem
    channel_ae = model.channels
    start_time = time.time()
    BERs = np.zeros_like(SNRs, dtype=float)
    # Computations
    for id_SNR in range(len(SNRs)):
        channel_ae.set_SNR_dB(SNRs[id_SNR], float(code_rate), modem.Es)
        bit_send = 0
        bit_err = 0
        rate = Fraction(code_rate).limit_denominator(100)
        divider = (Fraction(1, modem.num_bits_symbol * channel_ae.nb_tx) * 1 / rate).denominator
        send_chunk = args.send_chunk
        send_chunk = max(divider, send_chunk // divider * divider)

        receive_size = channel_ae.nb_tx * modem.num_bits_symbol
        ber = 0.0
        ber, output, X_train, fwd_noise = model(id_SNR, ber, args.send_chunk, receive_size)
        BERs[id_SNR] = ber / send_chunk
    if verbose:
        print('====> Test set for SNR', SNRs[0],
        'with ber ', float(BERs[0])
        )

    report_ber = float(BERs[0])
    # report_bler = float(test_bler)
    report_loss = 0

    return  report_loss,BERs
    # return report_ber


# def test((model, channel_ae, SNRs, code_rate, args, use_cuda=False, verbose=True):
def test(model, SNRS,  code_rate, filename, args, block_len='default', use_cuda=False):
    device = torch.device("cuda" if use_cuda else "cpu")
    model.eval()
    # snr_interval = (args.snr_test_end - args.snr_test_start) * 1.0 / (args.snr_points - 1)
    # SNRS = [snr_interval * item + args.snr_test_start for item in range(args.snr_points)]
    test_ber = 0.0
    bit_err = 0.0
    data_file = open(filename, 'a')
    modem = model.modem
    channel_ae = model.channels
    start_time = time.time()
    BERs = np.zeros_like(SNRS, dtype=float)
    # Computations
    for id_SNR in range(len(SNRS)):
        channel_ae.set_SNR_dB(SNRS[id_SNR], float(code_rate), modem.Es)
        bit_send = 0
        bit_err = 0
        rate = Fraction(code_rate).limit_denominator(100)
        divider = (Fraction(1, modem.num_bits_symbol * channel_ae.nb_tx) * 1 / rate).denominator
        send_chunk = args.send_chunk
        send_chunk = max(divider, send_chunk // divider * divider)

        receive_size = channel_ae.nb_tx * modem.num_bits_symbol
        ber = 0.0
        ber, output, X_train, fwd_noise = model(id_SNR, ber, args.send_chunk, receive_size)
        BERs[id_SNR] = ber / send_chunk
        data_file.write(str(SNRS[id_SNR]) + ' ' + str(BERs[id_SNR]) + "\n")

    data_file.close()
    print('final results on SNRs ', SNRS)
    print('BER', BERs)
    # compute adjusted SNR. (some quantization might make power!=1.0)
    # enc_power = 0.0
    # with torch.no_grad():
    #     for idx in range(num_test_batch):
    #         X_test = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)
    #         X_test = X_test.to(device)
    #         X_code = model.enc(X_test)
    #         enc_power += torch.std(X_code)
    # enc_power /= float(num_test_batch)
    # print('encoder power is', enc_power.item())
    # adj_snrs = [snr_sigma2db(snr_db2sigma(item) / enc_power) for item in snrs]
    # print('adjusted SNR should be', adj_snrs)
