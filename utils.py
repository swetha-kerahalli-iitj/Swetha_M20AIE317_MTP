import torch
import numpy as np
import math
import torch.nn.functional as F

from numpy.random import standard_normal
# import commpy.modulation as mod
# import commpy.channels as chan
# from commpy.modulation import QAMModem, kbest, best_first_detector
from pyphysim.modulators.fundamental import BPSK, QAM, QPSK, Modulator
# from pyphysim.simulations import Result, SimulationResults, SimulationRunner
from pyphysim.util.conversion import dB2Linear
from pyphysim.util.misc import pretty_time, randn_c, count_bit_errors
from pyphysim.channels import fading,fading_generators


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

def get_modem(mod_type=16):
    if mod_type== 'QAM16':
        return QAM(16)
    elif mod_type == 'QAM64':
        return QAM(64)
    else:
        return BPSK()

# def simulate_noise(mod, SNR,data,modulated_data,args,simulate):
#     """Return the symbol error rate"""
#     symbol_error_rate = 0.0
#     # data = np.random.randint(0, modulator.M, size=num_symbols)
#     # modulated_data = modulator.modulate(data)
#
#     # Noise vector
#     EbN0_linear = dB2Linear(SNR)
#     snr_linear = EbN0_linear * math.log2(mod.M)
#     noise_power = 1 / snr_linear
#     num_symbols = len(data)
#     n = math.sqrt(noise_power) * randn_c(num_symbols)
#     # received_data_qam = su_channel.corrupt_data(modulated_data_qam)
#     if simulate=='Rayleigh':
#         # Rayleigh channel
#         h = randn_c(modulated_data.size)
#
#         # Receive the corrupted data
#         received_data = h * modulated_data + n
#
#         # Equalization
#         received_data /= h
#     elif simulate=='Rician':
#         K_dB = args.code_rate_k  # K factor in dB
#         K = 10 ** (K_dB / 10)  # K factor in linear scale
#         mu = math.sqrt(K / (2 * (K + 1)))  # mean
#         sigma = math.sqrt(1 / (2 * (K + 1)))  # sigma
#         h = (sigma * standard_normal(num_symbols) + mu) + 1j * (sigma * standard_normal(num_symbols) + mu)
#         # Receive the corrupted data
#         received_data = h * modulated_data + n
#
#         # Equalization
#         received_data /= h
#     else:
#         # Receive the corrupted data
#         received_data = modulated_data + n
#
#     demodulated_data = mod.demodulate(received_data)
#     num_bit_errors = count_bit_errors(data, demodulated_data)
#     return num_bit_errors/num_symbols

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

def generate_noise_SNR(SNR,noise_shape,args,noise_type="AWGN",coderate_k=1,mod_type="QPSK16"):
    mod = get_modem(mod_type)
    input_msg = np.random.randint(0, mod.M, size=noise_shape[0] * noise_shape[1] * noise_shape[2])

    modulated_bits = mod.modulate(input_msg)

    # input_array = np.random.uniform(0,2,len(symbs))
    #
    # input_raleigh = symbs + input_array
    # input_raleigh = input_array

    EbN0_linear = dB2Linear(SNR)
    snr_linear = EbN0_linear * math.log2(mod.M)
    noise_power = 1 / snr_linear
    num_symbols = len(input_msg)
    n = math.sqrt(noise_power) * randn_c(num_symbols)

    # received_data_qam = su_channel.corrupt_data(modulated_data_qam)
    if noise_type== 'Rayleigh':
        # Rayleigh channel
        h = randn_c(modulated_bits.size)

        # Receive the corrupted data
        received_data = h * modulated_bits + n


        # Equalization
        received_data /= h
    elif noise_type == 'Rician':
        K_dB = coderate_k # K factor in dB
        K = 10 ** (K_dB / 10)  # K factor in linear scale
        mu = math.sqrt(K / (2 * (K + 1)))  # mean
        sigma = math.sqrt(1 / (2 * (K + 1)))  # sigma
        h = (sigma * standard_normal(num_symbols) + mu) + 1j * (sigma * standard_normal(num_symbols) + mu)
        # Receive the corrupted data
        received_data = h * modulated_bits + n

        # Equalization
        received_data /= h
    else:
        # Receive the corrupted data
        n = snr_db2sigma(SNR) *randn_c(num_symbols)
        received_data = modulated_bits + n
    # channel_output = np.random.rayleigh(abs(SNR),size=noise_shape) + resized_input
    resized_noise_output = torch.from_numpy(np.array(received_data).reshape(noise_shape))

    return resized_noise_output,modulated_bits,input_msg

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