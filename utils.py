from fractions import Fraction

import torch
import numpy as np
import math
import torch.nn.functional as F
import commpy.modulation as mod
import commpy.channels as chan
from commpy.modulation import QAMModem, kbest, best_first_detector

def errors_ber(y_true, y_pred, positions = 'default'):
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    myOtherTensor = torch.ne(torch.round(y_true), torch.round(y_pred)).float()
    if positions == 'default':
        res = sum(sum(myOtherTensor))/(myOtherTensor.shape[0]*myOtherTensor.shape[1])
    else:
        res = torch.mean(myOtherTensor, dim=0).type(torch.FloatTensor)
        for pos in positions:
            res[pos] = 0.0
        res = torch.mean(res)
    return res

def errors_ber_list(y_true, y_pred):
    block_len = y_true.shape[1]
    y_true = y_true.view(y_true.shape[0], -1)
    y_pred = y_pred.view(y_pred.shape[0], -1)

    myOtherTensor = torch.ne(torch.round(y_true), torch.round(y_pred))
    res_list_tensor = torch.sum(myOtherTensor, dim = 1).type(torch.FloatTensor)/block_len

    return res_list_tensor


def errors_ber_pos(y_true, y_pred, discard_pos = []):
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    myOtherTensor = torch.ne(torch.round(y_true), torch.round(y_pred)).float()

    tmp =  myOtherTensor.sum(0)/myOtherTensor.shape[0]
    res = tmp.squeeze(1)
    return res

def code_power(the_codes):
    the_codes = the_codes.cpu().numpy()
    the_codes = np.abs(the_codes)**2
    the_codes = the_codes.sum(2)/the_codes.shape[2]
    tmp =  the_codes.sum(0)/the_codes.shape[0]
    res = tmp
    return res

def errors_bler(y_true, y_pred, positions = 'default'):

    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    decoded_bits = torch.round(y_pred)
    X_test       = torch.round(y_true)
    tp0 = (abs(decoded_bits-X_test)).view([X_test.shape[0],X_test.shape[1]])
    tp0 = tp0.cpu().numpy()

    if positions == 'default':
        bler_err_rate = sum(np.sum(tp0,axis=1)>0)*1.0/(X_test.shape[0])
    else:
        for pos in positions:
            tp0[:, pos] = 0.0
        bler_err_rate = sum(np.sum(tp0,axis=1)>0)*1.0/(X_test.shape[0])

    return bler_err_rate

# note there are a few definitions of SNR. In our result, we stick to the following SNR setup.
def snr_db2sigma(train_snr):
    return 10**(-train_snr*1.0/20)

def snr_sigma2db(train_sigma):
    try:
        return -20.0 * math.log(train_sigma, 10)
    except:
        return -20.0 * torch.log10(train_sigma)
def get_modem(mod_type='QAM_16'):
    if mod_type == 'QAM_16':
        return mod.QAMModem(16)

def awgn_ber(input,mod_bits,SNR,noise_shape,mod):
    channel_output = chan.awgn(input, SNR)
    send_chunk = noise_shape[0]
    decoded_bits = mod.demodulate(channel_output, 'hard')
    # calculate number of error frames
    number_chunks_per_send = noise_shape[1]
    bit_err = 0
    chunk_loss = 0
    chunk_count = 0
    total_tx_send = 0
    # for i in range(number_chunks_per_send):
    #     errors = np.bitwise_xor(mod_bits[send_chunk * i:send_chunk * (i + 1)],
    #                             decoded_bits[send_chunk * i:send_chunk * (i + 1)].astype(int)).sum()
    #     bit_err += errors
    #     chunk_loss += 1 if errors > 0 else 0
    bit_err = 1 - sum(
        decoded_bits == mod_bits) / decoded_bits.size

    return bit_err

    #     chunk_count += number_chunks_per_send
    #     total_tx_send += 1
    # return  bit_err.sum() / (total_tx_send * send_chunk)
def receiver(mod,y, h, constellation, noise_var):
        return mod.demodulate(kbest(y, h, constellation, 16), 'hard')
def rayleigh_ber(mod_input_bits ,msg,SNR,noise_shape,mod,code_rate):
    RayleighChannel = chan.MIMOFlatChannel(noise_shape[2], noise_shape[2])
    RayleighChannel.uncorr_rayleigh_fading(complex)
    RayleighChannel.set_SNR_dB(SNR, float(code_rate), mod.Es)
    channel_output = RayleighChannel.propagate(mod_input_bits)
    # channel_output = np.random.rayleigh(abs(SNR),size=input.shape)+input
    send_chunk = noise_shape[0]
    if type(code_rate) is float:
        code_rate = Fraction(code_rate).limit_denominator(100)
    rate = code_rate
    divider = (Fraction(1, mod.num_bits_symbol * RayleighChannel.nb_tx) * 1 / code_rate).denominator
    send_chunk = max(divider, send_chunk // divider * divider)

    receive_size = RayleighChannel.nb_tx * mod.num_bits_symbol
    # Deals with MIMO channel
    if isinstance(RayleighChannel, chan.MIMOFlatChannel):
        nb_symb_vector = len(channel_output)
        received_msg = np.empty(int(math.ceil(len(msg) / float(rate))))
        for i in range(nb_symb_vector):
            received_msg[receive_size * i:receive_size * (i + 1)] = \
                receiver(mod,channel_output[i], RayleighChannel.channel_gains[i],
                             mod.constellation, RayleighChannel.noise_std ** 2)
    else:
        received_msg = receiver(mod,channel_output, RayleighChannel.channel_gains,
                                    mod.constellation, RayleighChannel.noise_std ** 2)
    # Count errors
    decoder = lambda msg: msg
    decoded_bits = decoder(received_msg)
    # calculate number of error frames
    number_chunks_per_send = noise_shape[1]
    bit_err = 0
    chunk_loss = 0
    chunk_count = 0
    total_tx_send = 0
    for i in range(number_chunks_per_send):
        errors = np.bitwise_xor(msg[send_chunk * i:send_chunk * (i + 1)],
                                decoded_bits[send_chunk * i:send_chunk * (i + 1)].astype(int)).sum()
        bit_err += errors
        chunk_loss += 1 if errors > 0 else 0

        chunk_count += number_chunks_per_send
        total_tx_send += 1
    return  bit_err.sum() / (total_tx_send * send_chunk)
def get_theo_ber(SNR,M=16):
    k = math.log2(M);
    SNRLin = 10 ** (SNR / 10);
    gamma_c = SNRLin * k;

    if M == 4:
        # 4 - QAM
        ber = 1 / 2 * (1 - math.sqrt(gamma_c / k/ (1 + gamma_c / k)));
    elif M == 16 :
        # 16 - QAM
        ber = 3 / 8 * (1 - math.sqrt(2 / 5 * gamma_c / k / (1 + 2 / 5 * gamma_c / k)));
    elif M == 64:
        # 64 - QAM
        ber = 7 / 24 * (1 - math.sqrt(1 / 7 * gamma_c / k/ (1 + 1 / 7 * gamma_c / k)));
    return ber
def gen_mod(input_msg,noise_shape,mod):
    symbs = mod.modulate(input_msg)
    input_array = np.random.uniform(0, 2, len(symbs))
    output = symbs +input_array
    resized_output = torch.from_numpy(np.array(output).reshape(noise_shape)).real.float()
    return  resized_output

def generate_Rayleigh_noise_SNR(SNR,noise_shape,args,mod_type ='QAM_16',code_rate=1/3):
    input_msg = np.random.choice((0, 1), args.batch_size * args.block_len * args.code_rate_n * 4)
    msg = input_msg
    mod = get_modem(mod_type)
    symbs = mod.modulate(msg)

    input_array = np.random.uniform(0,2,len(symbs))

    input_raleigh = symbs + input_array
    # input_raleigh = input_array

    # resized_symbs = torch.from_numpy(np.array(input_raleigh).reshape(noise_shape))
    RayleighChannel = chan.MIMOFlatChannel(noise_shape[2],noise_shape[2])
    RayleighChannel.uncorr_rayleigh_fading(complex)
    RayleighChannel.set_SNR_dB(SNR, float(code_rate), mod.Es)
    channel_output = RayleighChannel.propagate(input_raleigh)
    resized_input = np.array(input_raleigh).reshape(noise_shape)
    # channel_output = np.random.rayleigh(abs(SNR),size=noise_shape) + resized_input
    resized_output = np.array(channel_output).reshape(noise_shape)
    # resized_output = generate_noise(noise_shape,args)

    return torch.from_numpy(resized_output),input_raleigh,msg
    # return resized_output, torch.from_numpy(resized_input)

def generate_noise(noise_shape, args, test_sigma = 'default', snr_low = 0.0, snr_high = 0.0, mode = 'encoder'):
    # SNRs at training
    if test_sigma == 'default':
        this_sigma_low = snr_db2sigma(snr_low)
        this_sigma_high= snr_db2sigma(snr_high)
        # mixture of noise sigma.
        this_sigma = (this_sigma_low - this_sigma_high) * torch.rand(noise_shape) + this_sigma_high

    else:
        this_sigma = snr_db2sigma(test_sigma)

    # SNRs at testing
    if args.channel == 'awgn':
        fwd_noise  = this_sigma * torch.randn(noise_shape, dtype=torch.float)

    elif args.channel == 't-dist':
        fwd_noise  = this_sigma * torch.from_numpy(np.sqrt((args.vv-2)/args.vv) * np.random.standard_t(args.vv, size = noise_shape)).type(torch.FloatTensor)

    elif args.channel == 'radar':
        add_pos     = np.random.choice([0.0, 1.0], noise_shape,
                                       p=[1 - args.radar_prob, args.radar_prob])

        corrupted_signal = args.radar_power* np.random.standard_normal( size = noise_shape ) * add_pos
        fwd_noise = this_sigma * torch.randn(noise_shape, dtype=torch.float) +\
                    torch.from_numpy(corrupted_signal).type(torch.FloatTensor)

    else:
        # Unspecific channel, use AWGN channel.
        fwd_noise  = this_sigma * torch.randn(noise_shape, dtype=torch.float)

    return fwd_noise

def customized_loss(output, X_train, args, size_average = True, noise = None, code = None):
    output = torch.clamp(output, 0.0, 1.0)
    if size_average == True:
        loss = F.binary_cross_entropy(output, X_train)
    else:
        return [F.binary_cross_entropy(item1, item2) for item1, item2 in zip(output, X_train)]

    return loss
class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, args):

        ctx.save_for_backward(inputs)
        ctx.args = args

        x_lim_abs  = args.enc_value_limit
        x_lim_range = 2.0 * x_lim_abs
        x_input_norm =  torch.clamp(inputs, -x_lim_abs, x_lim_abs)

        if args.enc_quantize_level == 2:
            outputs_int = torch.sign(x_input_norm)
        else:
            outputs_int  = torch.round((x_input_norm +x_lim_abs) * ((args.enc_quantize_level - 1.0)/x_lim_range)) * x_lim_range/(args.enc_quantize_level - 1.0) - x_lim_abs

        return outputs_int

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.args.enc_clipping in ['inputs', 'both']:
            input, = ctx.saved_tensors
            grad_output[input>ctx.args.enc_value_limit]=0
            grad_output[input<-ctx.args.enc_value_limit]=0

        if ctx.args.enc_clipping in ['gradient', 'both']:
            grad_output = torch.clamp(grad_output, -ctx.args.enc_grad_limit, ctx.args.enc_grad_limit)

        if ctx.args.train_channel_mode not in ['group_norm_noisy', 'group_norm_noisy_quantize']:
            grad_input = grad_output.clone()
        else:
            # Experimental pass gradient noise to encoder.
            grad_noise = snr_db2sigma(ctx.args.fb_noise_snr) * torch.randn(grad_output[0].shape, dtype=torch.float)
            ave_temp   = grad_output.mean(dim=0) + grad_noise
            ave_grad   = torch.stack([ave_temp for _ in range(ctx.args.batch_size)], dim=2).permute(2,0,1)
            grad_input = ave_grad + grad_noise

        return grad_input, None