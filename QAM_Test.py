import math

import numpy as np
from matplotlib import pyplot as plt

from pyphysim.modulators.fundamental import BPSK, QAM, QPSK, Modulator
# from pyphysim.simulations import Result, SimulationResults, SimulationRunner
from pyphysim.util.conversion import dB2Linear
from pyphysim.util.misc import pretty_time, randn_c, count_bit_errors

from utils import simulate_noise

# def simulate_noise(mod, SNR,data,modulated_data,simulate_with_rayleigh=False):
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
#
#     # received_data_qam = su_channel.corrupt_data(modulated_data_qam)
#     if simulate_with_rayleigh:
#         # Rayleigh channel
#         h = randn_c(modulated_data.size)
#
#         # Receive the corrupted data
#         received_data = h * modulated_data + n
#
#         # Equalization
#         received_data /= h
#
#     else:
#         # Receive the corrupted data
#         received_data = modulated_data + n
#
#     demodulated_data = mod.demodulate(received_data)
#     symbol_error_rate = sum( demodulated_data != data)
#     num_bit_errors = count_bit_errors(data, demodulated_data)
#     return num_bit_errors/num_symbols

if __name__ == '__main__':
    SNRS = np.linspace(-5, 15, 9)
    noise_shape = (100, 100, 3)
    mod = QPSK()
    # input_msg = np.random.choice((0, 1), noise_shape[0] * noise_shape[1] * noise_shape[2] * 4)
    msg = np.random.randint(0, mod.M, size=noise_shape[0] * noise_shape[1] * noise_shape[2] * 4)
    symbs = mod.modulate(msg)

    input_array = np.random.uniform(0, 2, len(symbs))

    input_raleigh = symbs + input_array
    idx = 0
    awgn_bers = np.zeros_like(SNRS, dtype=float)
    rayleigh_bers = np.zeros_like(SNRS, dtype=float)

    for SNR in SNRS:
        # noise_shape = (100, 100, 3)
        rayleigh_ber = 0.0
        awgn_ber = 0.0
        iteration = 10
        for iter in range(iteration):
            # print('executing iteration {} for SNR {}'.format(iter,SNR))
            SNR_dB = 20
            snr_linear = dB2Linear(SNR_dB)
            noise_power = 1 / snr_linear
            awgn_ber += simulate_noise(mod,SNR,msg,symbs,False)
            rayleigh_ber += simulate_noise(mod, SNR, msg, symbs,True)

        awgn_bers[idx] = awgn_ber/iteration
        rayleigh_bers[idx] = rayleigh_ber / iteration
        idx += 1
    # Now let's plot the results
    fig, ax = plt.subplots(figsize=(5, 5))
    print('SNRS',SNRS)
    print ('theo',mod.calcTheoreticalBER(SNRS))
    print('awgn', awgn_bers)
    print('rayleigh_bers', rayleigh_bers)
    ax.semilogy(SNRS, mod.calcTheoreticalBER(SNRS), "--", label="Theoretical")
    ax.semilogy(SNRS,awgn_bers,
                label="Simulated awgn")
    ax.semilogy(SNRS, rayleigh_bers,
                label="Simulated rayleigh")
    ax.set_title("QAM-16 Symbol Error Rate ")
    ax.set_ylabel("Symbol Error Rate")
    ax.set_xlabel("SNR (dB)")
    ax.legend()
    plt.show()

