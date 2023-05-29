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
        received_codes = codes + fwd_noise

        if self.args.rec_quantize:
            myquantize = MyQuantize.apply
            received_codes = myquantize(received_codes, self.args)

        resized_decoded_bits = received_codes
        x_dec = self.dec(resized_decoded_bits)

        return x_dec, codes