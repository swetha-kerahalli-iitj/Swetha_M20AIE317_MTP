import math

import numpy as np
import torch
from numpy.random import standard_normal
from pyphysim.util.misc import randn_c

from utils import STEQuantize as MyQuantize, generate_decode


class Channel_AE(torch.nn.Module):
    def __init__(self, args, enc, dec):
        super(Channel_AE, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        self.enc = enc
        self.dec = dec

    def forward(self, input, fwd_noise,noise_type, mod,SNR, noise_shape,coderate_k,  mod_type):
        codes = self.enc(input)
        codesshape = codes.shape
        # received_codes = codes + fwd_noise
        if noise_type == "Rayleigh":
            h = torch.from_numpy(randn_c(codesshape[0]*codesshape[1]*codesshape[2])).reshape(codesshape)
            received_codes = h * codes + fwd_noise
            received_codes /= h
        elif noise_type == "Rician":
            K_dB = noise_shape[2] # K factor in dB
            K = 10 ** (K_dB / 10)  # K factor in linear scale
            mu = math.sqrt(K / (2 * (K + 1)))  # mean
            sigma = math.sqrt(1 / (2 * (K + 1)))  # sigma
            h = torch.from_numpy((sigma * standard_normal(noise_shape[0]*noise_shape[1]*noise_shape[2]) + mu) + 1j * (
                        sigma * standard_normal(noise_shape[0]*noise_shape[1]*noise_shape[2]) + mu)).reshape(noise_shape)
            received_codes = h * codes + fwd_noise
            received_codes /= h
        else:
            # print('default AWGN channel')
            received_codes = codes + fwd_noise

        if self.args.rec_quantize:
            myquantize = MyQuantize.apply
            received_codes = myquantize(received_codes, self.args)
        # resized_decoded_bits = received_codes
        ber,decoded_bits =  generate_decode(received_codes, input,mod,SNR, noise_shape,  mod_type,ber_reqd = 0)
        resized_decoded_bits = torch.from_numpy(
        np.array(decoded_bits[0:noise_shape[1] * noise_shape[0] * noise_shape[2]]).reshape(noise_shape))
        x_dec = self.dec(resized_decoded_bits)

        return x_dec, codes