import math

import numpy as np
from matplotlib import pyplot as plt

from pyphysim.modulators.fundamental import BPSK, QAM, QPSK, Modulator
# from pyphysim.simulations import Result, SimulationResults, SimulationRunner
from pyphysim.util.conversion import dB2Linear
from pyphysim.util.misc import pretty_time, randn_c
def simulate_awgn(mod, SNR,data,modulated_data):
    """Return the symbol error rate"""
    symbol_error_rate = 0.0
    # data = np.random.randint(0, modulator.M, size=num_symbols)
    # modulated_data = modulator.modulate(data)

    # Noise vector
    snr_linear = dB2Linear(SNR)
    noise_power = 1 / snr_linear
    num_symbols = len(data)
    n = math.sqrt(noise_power) * randn_c(num_symbols)

    # received_data_qam = su_channel.corrupt_data(modulated_data_qam)
    received_data = modulated_data + n

    demodulated_data = mod.demodulate(received_data)
    symbol_error_rate = sum( demodulated_data != data)
    return symbol_error_rate

if __name__ == '__main__':
    SNRS = np.linspace(-5, 15, 9)
    noise_shape = (2, 2, 3)
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
        ber = 0.0
        iteration = 10000
        for iter in range(iteration):
            SNR_dB = 20
            snr_linear = dB2Linear(SNR_dB)
            noise_power = 1 / snr_linear
            ber += simulate_awgn(mod,SNR,msg,symbs)

        awgn_bers[idx] = ber/iteration
        idx += 1
    # Now let's plot the results
    fig, ax = plt.subplots(figsize=(5, 5))
    print('SNRS',SNRS)
    print ('theo',mod.calcTheoreticalBER(SNRS))
    print('awgn', awgn_bers)
    ax.semilogy(SNRS, mod.calcTheoreticalBER(SNRS), "--", label="Theoretical")
    ax.semilogy(SNRS,awgn_bers,
                label="Simulated")
    ax.set_title("QPSK Symbol Error Rate (AWGN channel)")
    ax.set_ylabel("Symbol Error Rate")
    ax.set_xlabel("SNR (dB)")
    ax.legend()
    plt.show()

