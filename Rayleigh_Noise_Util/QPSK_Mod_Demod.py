import matplotlib.pyplot as plt
import numpy as np
import torch


class QPSK_Mod_Demod(torch.nn.Module):
    def __init__(self, args, modetype):
        super(QPSK_Mod_Demod, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        self.modetype = modetype
        self.SNR =args.SNR
        self.nb = args.nb
        self.qpsk_mode_type = args.qpsk_mode_type
        self.plot_file = args.plot_file_enc
        self.T = args.T


    def QPSK_Modulation(self,input):
            # T - Baseband signal width, which is frequency
            # nb - Define the number of bits transmitted
            t=input
            N = t.shape[0]*t.shape[1]*t.shape[2]
            # self.nb = (int) (len(t)/200)# (197136,)198916- 197136 = 1000, 200000-198916
            # delta_T = self.T / (self.nb * (self.qpsk_mode_type))  # sampling interval
            self.nb =(int) (N/200)
            delta_T = 1 #self.T / (self.nb * (self.qpsk_mode_type / N))  # sampling interval
            fs = 1 / delta_T  # Sampling frequency
            fc = 10 / self.T  # Carrier frequency
            # SNR - Signal to noise ratio
            # t = np.arange(0, self.nb * self.T, delta_T)

            # print("DeltaT =",delta_T,
            #       "fs =",fs,
            #       "N =", N,
            #       "nb =",self.nb)
            # Generate baseband signal
            data =  np.random.random_integers(0,1, (t.shape[0],t.shape[1],t.shape[2]*2))
            # data = [1 if x > 0.5 else 0 for x in np.random.random_integers(0,1, t.shape)]
            # Call the random function to generate any 1*nb matrix from 0 to 1, which is greater than 0.5 as 1 and less than 0.5 as 0
            data0 = np.zeros(data.shape)  # Create a 1*nb/delta_T zero matrix
            # for q in range(self.nb):
            #     data0 += [data[q]] * int(1 / delta_T)  # Convert the baseband signal into the corresponding waveform signal
            data0 = np.array(data0) + (data * (int(1 / delta_T)))
            # Modulation signal generation
            data1 = np.zeros(data.shape) # Create a 1*nb/delta_T zero matrix
            datanrz = np.array(data) * 2 - 1  # Convert the baseband signal into a polar code, mapping
            # for q in range(self.nb):
            #     data1 += [datanrz[q]] * int(1 / delta_T)  # Change the polarity code into the corresponding waveform signal
            data1 += (datanrz * int(1 / delta_T))
            idata = datanrz[:,:,0:(
                    self.nb - 1):2]  # Serial and parallel conversion, separate the odd and even bits, the interval is 2, i is the odd bit q is the even bit
            qdata = datanrz[:,:,1:self.nb:2]
            ich = np.zeros(idata.shape)  # Create a 1*nb/delta_T/2 zero matrix to store parity data later
            qch = np.zeros(qdata.shape)
            # for i in range(int(self.nb /2)):
            #     ich += [idata[i]] * int(1 / delta_T)  # Odd bit symbols are converted to corresponding waveform signals
            #     qch += [qdata[i]] * int(1 / delta_T)  # Even bit symbols are converted to corresponding waveform signals
            ich += idata * int(1 / delta_T)
            qch += qdata * int(1 / delta_T)
            a = []  # Cosine function carrier
            b = []  # Sine function carrier
            # for j in range(int(N / 2)):
            #     a.append(np.math.sqrt(2 / self.T) * np.math.cos(2 * np.math.pi * fc * t[j]))  # Cosine function carrier
            #     b.append(np.math.sqrt(2 / self.T) * np.math.sin(2 * np.math.pi * fc * t[j]))  # Sine function carrier
            input_cosine = 2 * np.math.pi * fc * t
            # input_sine = 2 * np.math.pi * fc * t
            input_cos = np.cos(input_cosine)
            input_sin = np.sin(input_cosine)
            a = np.math.sqrt(2 / self.T) * input_cos
            b = np.math.sqrt(2 / self.T) * input_sin
            # ibase = np.array(data0[0:(nb - 1):2]) * np.array(a)
            idata1 = np.array(ich) * np.array(
                a)  # Odd-digit data is multiplied by the cosine function to get a modulated signal
            qdata1 = np.array(qch) * np.array(
                b)  # Even-digit data is multiplied by the cosine function to get another modulated signal
            s = idata1 + qdata1  # Combine the odd and even data, s is the QPSK modulation signal
            if self.plot_file != '':
                sample_size = N / (self.qpsk_mode_type / 2)
                plt.figure(figsize=(20, 12))
                plt.subplot(4, 1, 1)
                plt.plot(np.sin(t))
                plt.title('Pure', fontsize=10)
                plt.subplot(4, 1, 2)
                plt.plot(idata1)
                plt.title('In-phase branch I', fontsize=10)
                plt.axis([0, sample_size, -3, 3])
                plt.subplot(4, 1, 3)
                plt.plot(qdata1)
                plt.title('Orthogonal branch Q', fontsize=10)
                plt.axis([0, sample_size, -3, 3])
                plt.subplot(4, 1, 4)
                plt.plot(s)
                plt.title('Modulated signal', fontsize=10)
                plt.axis([0, sample_size, -3, 3])
                plt.savefig(self.plot_file)
                plt.show()

            return s


    def Encoder(self,input):
        if self.modetype == 'encoder':
            qpsk_mod = self.QPSK_Modulation(input)
        return  qpsk_mod

    def forward(self, input, fwd_noise,show_plot= False,plot_file=''):
        qpsk_mode_type = 8
        if self.modetype == 'encoder':
            qpsk_mod = self.QPSK_Modulation(input)

        # # Convolve sinusoidal waveform with Rayleigh Fading channel
        # y3 = np.convolve(fwd_noise, qpsk_mod)
        y3 = fwd_noise+qpsk_mod
        if show_plot:
            sample_size = len(qpsk_mod)/qpsk_mode_type
            plt.figure(figsize=(30, 12))
            plt.subplot(5, 1, 1)
            plt.plot(np.sin(input))
            plt.title('Pure sine wave signal', fontsize=10)
            plt.axis([0, sample_size, -3, 3])
            plt.subplot(5, 1, 2)
            plt.plot(qpsk_mod)
            plt.title('QPSK Modulated wave signal', fontsize=10)
            plt.axis([0, sample_size, -3, 3])
            plt.subplot(5, 1,3)
            plt.plot(fwd_noise)
            plt.title('Rayleigh Noise', fontsize=10)
            plt.axis([0, sample_size, -3, 3])
            plt.subplot(5, 1,4)
            plt.plot(y3)
            plt.title('Convolved sine wave signal', fontsize=10)
            plt.axis([0, sample_size, -100, 100])
            plt.tight_layout()
            plt.savefig(plot_file)
            plt.show()
        return y3,qpsk_mod