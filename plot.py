import os

import numpy
import numpy as np
from matplotlib import pyplot as plt


def get_plots(plot_path,filename):
    file_data = np.loadtxt(filename,dtype = str)
    semlogy,legend,blocklens,coderates,mod_types =[],[],[],[],[]

    test_file_path = os.path.join(plot_path, 'All_Noise_SNR_BER'+'.png')
    # plt.figure(figsize=(15,15))
    for test_row in file_data:
        blocklen ,coderate_k,coderate_n,coderate,mod_type,filename,testfilename =test_row
        blocklens.append(blocklen)
        mod_types.append(mod_type)
        coderate_rounded = round(float(coderate),2)
        coderates.append('{}/{}'.format(coderate_k,coderate_n))

        # data = np.loadtxt(filename)
        test_data = np.loadtxt(testfilename).T
        snrs = test_data[3,:]
        awgn_bers= test_data[4,:]
        ray_bers= test_data[5,:]
        rici_bers= test_data[6,:]

        legend.append('AWGN_{}_{}/{}_{}'.format(blocklen,coderate_k,coderate_n,mod_type))
        legend.append('Rayleigh_{}_{}/{}_{}'.format(blocklen,coderate_k,coderate_n,mod_type))
        legend.append('Rician_{}_{}/{}_{}'.format(blocklen,coderate_k,coderate_n,mod_type))
        plt.semilogy(snrs,awgn_bers,'--',snrs,ray_bers,'x-',snrs,rici_bers,'o-')
        # plt.title(legend)

    # plt.semilogy(list(semlogy))
    plt.grid()
    plt.xticks(snrs)
    plt.title('With blocklen - {}/ coderates {}/ modulation types {} snr-ber'.format(np.unique(blocklens), np.unique(coderates), np.unique(mod_types)))
    plt.legend(legend,loc='center right')
    plt.savefig(test_file_path, format='png', bbox_inches='tight',
                dpi=1200)
    plt.show()