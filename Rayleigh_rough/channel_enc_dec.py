from math import log2,log10

import torch

# import commpy.links as lk
from moddemod_qpsk_links import LinkModel
import commpy.channelcoding.convcode as cc
from commpy.channelcoding.ldpc import get_ldpc_code_params, triang_ldpc_systematic_encode, ldpc_bp_decode
from commpy.modulation import QAMModem, kbest, best_first_detector


class Channel_AE(torch.nn.Module):
    def __init__(self, args, channels, modem, code_rate):
        super(Channel_AE, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        self.channels = channels
        self.modem = modem
        self.code_rate = code_rate
        self.linkmodel = LinkModel(modem.modulate, channels, self.receiver,
                            modem.num_bits_symbol, modem.constellation, modem.Es)


    # Modulation function
    def receiver(self,y, h, constellation, noise_var):
        return self.modem.demodulate(kbest(y, h, constellation, 16), 'hard')

    def forward(self, SNR,ber,send_chunk, receive_size):
        ber = self.linkmodel.link_performance_full_metrics_iteration( SNR,ber,send_chunk, receive_size,code_rate=self.code_rate)
        output = torch.from_numpy(self.channels.unnoisy_output + self.channels.noises)
        X_train = torch.from_numpy(self.channels.unnoisy_output)
        fwd_noise = torch.from_numpy(self.channels.noises)

        return ber,output,X_train,fwd_noise