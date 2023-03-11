import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from numpy import random

# import matlab.engine

BASE_PATH = r'C:\WorkSpace\Swetha_M20AIE317_MTP'
LOG_PATH = os.path.join(BASE_PATH, r'logs_new')
DATA_PATH = os.path.join(BASE_PATH, r'data_new')
MODEL_PATH = os.path.join(BASE_PATH, r'model_new')
for path in (LOG_PATH, DATA_PATH, MODEL_PATH):
    if not os.path.isdir(path):
        os.makedirs(path)
attnFilename = []


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


# import Rayleigh.rician
if __name__ == '__main__':


    filename = r'C:\WorkSpace\Swetha_M20AIE317_MTP\data_test123\attention_data_awgn_lr_0.01_D10_10000_20230310-153356.txt'
    test_filename = r'C:\WorkSpace\Swetha_M20AIE317_MTP\data_test123\attention_data_test_awgn_lr_0.01_D10_10000_20230310-153356.txt'
    data = np.loadtxt(filename, usecols=[0,2]).T
    test_data= np.loadtxt(test_filename, usecols=[0,1]).T
    snrs = test_data[0,:]
    ber = test_data[1,:]
    epoch =data[0,:]
    ber_epoch = data[1,:]
    plt.semilogy(epoch, ber_epoch, 'o-')
    plt.grid()
    plt.xlabel('Epoch)')
    plt.ylabel('Bit Error Rate')
    plt.legend('epoch-ber')
    plt.savefig('./attention_D_epcoh_' + str(1) + '.png', format='png', bbox_inches='tight',
                transparent=True, dpi=800)
    plt.show()


    plt.semilogy(snrs, ber, 'o-')
    plt.grid()
    plt.xlabel('Signal to Noise Ration (dB)')
    plt.ylabel('Bit Error Rate')
    plt.legend('SNR-ber')
    plt.savefig('./attention_D_snr_' + str(1) + '.png', format='png', bbox_inches='tight',
                transparent=True, dpi=800)
    plt.show()


    #s = Rice(values['-K'], values['-\hat{r}^2'], values['-\phi'])
    # noise_rand = torch.randn((10,10,3), dtype=torch.float)
    # noise = gen_reyleigh_246QPSK(noise_rand)
    # tfp.random.rayleigh(
    # bitrate = 16
    # r = np.random.rayleigh(size=(100,100,3))
    # (car1,car2) = CarrierWave(bitrate)
    # QPSk_Mod(bitrate,car1,car2)
    print(123)
    # eng = matlab.engine.start_matlab()
    # qpskMod = eng.comm.QPSKModulator
    # qpskDemod = eng.comm.QPSKDemodulator
    # rayleighchan = eng.comm.RicianChannel( 'SampleRate',10e3,
    # 'PathDelays',[0 ,1.5e-4],
    # 'AveragePathGains',[2 ,3],
    # 'NormalizePathGains',True,
    # 'MaximumDopplerShift',30,
    # 'DopplerSpectrum',{eng.doppler('Gaussian',0.6),eng.doppler('Flat')},
    # 'RandomStream','mt19937ar with seed',
    # 'Seed',22,
    # 'PathGainsOutputPort',True)
    # awgnChan = eng.comm.AWGNChannel(
    # NoiseMethod = 'Signal to noise ratio (SNR)')
    # errorCalc = eng.comm.ErrorRate
    # bitRate = 20e3;
    # M = 4
    # tx = eng.randi([0, M - 1], 100000, 1)
    #
    # qpskSig = qpskMod(tx);
    # fadedSig = rayleighchan(qpskSig);
    # eng.disp(fadedSig)
    #
    # SNR = (0,2,20)
    # numSNR = eng.length(SNR);
    # berVec = eng.zeros(3, numSNR);
    #
    # for n in range(numSNR):
    #     awgnChan.SNR = SNR(n);
    #     rxSig = awgnChan(fadedSig);
    #     rx = qpskDemod(rxSig);
    #     eng.reset(errorCalc)
    #     berVec[:,n] = errorCalc(tx, rx);
    #
    # BER = berVec[1,:]
    # BERtheory = eng.berfading(SNR, 'oqpsk', M, 1);
    # print('*********************************************')
    # print(SNR)
    # print('*********************************************')
    # print(BERtheory)
    #
    # eng.semilogy(SNR, BERtheory, 'b-', SNR, BER, 'r*');
    # eng.legend('Theoretical BER', 'Empirical BER');
    # eng.xlabel('SNR (dB)');
    # eng.ylabel('BER');
    # eng.title('Binary DPSK over Rayleigh Fading Channel');
    # eng.title('Binary QPSK over Rayleigh Fading Channel');
    #
    # eng.quit()

