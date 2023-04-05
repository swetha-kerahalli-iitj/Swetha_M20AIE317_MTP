import os

import numpy
import numpy as np
from matplotlib import pyplot as plt

def get_plots_all(plot_path,filename):
    file_data = np.loadtxt(filename, dtype=str)
    semlogy, legend, blocklens, coderates, mod_types = [], [], [], [], []

    test_file_path = os.path.join(plot_path, 'All_Noise_SNR_BER' + '.png')

    fig = plt.figure()
    ax = plt.subplot(111)
    for test_row in file_data:
        blocklen, coderate_k, coderate_n, coderate, mod_type, filename, testfilename = test_row
        blocklens.append(blocklen)
        mod_types.append(mod_type)
        coderate_rounded = round(float(coderate), 2)
        coderates.append('{}/{}'.format(coderate_k, coderate_n))

        # data = np.loadtxt(filename)
        test_data = np.loadtxt(testfilename).T
        snrs = test_data[3, :]
        LC_awgn_bers = test_data[4, :]
        LC_ray_bers = test_data[5, :]
        LC_rici_bers = test_data[6, :]
        awgn_bers = test_data[7, :]
        ray_bers = test_data[8, :]
        rici_bers = test_data[9, :]
        legend.append('LC_AWGN_{}_{}/{}_{}'.format(blocklen, coderate_k, coderate_n, mod_type))
        legend.append('LC_Rayleigh_{}_{}/{}_{}'.format(blocklen, coderate_k, coderate_n, mod_type))
        legend.append('LC_Rician_{}_{}/{}_{}'.format(blocklen, coderate_k, coderate_n, mod_type))
        legend.append('AWGN_{}_{}/{}_{}'.format(blocklen, coderate_k, coderate_n, mod_type))
        legend.append('Rayleigh_{}_{}/{}_{}'.format(blocklen, coderate_k, coderate_n, mod_type))
        legend.append('Rician_{}_{}/{}_{}'.format(blocklen, coderate_k, coderate_n, mod_type))
        ax.semilogy(snrs, LC_awgn_bers, '--', snrs, LC_ray_bers, 'x-', snrs, LC_rici_bers, 'o-',
                    snrs, awgn_bers, '-<', snrs, ray_bers, '->', snrs, rici_bers, '-|')

    plt.grid()
    plt.xticks(snrs)
    plt.title('With blocklen - {}/ coderates {}/ modulation types {} snr-ber'.format(np.unique(blocklens),
                                                                                     np.unique(coderates),
                                                                                     np.unique(mod_types)),loc='left')

    leg = ax.legend(legend,bbox_to_anchor=(1.04, 0.0, 0.3, 1), loc="upper left",ncol=3,fontsize='xx-small'
                    # ,borderaxespad=0, mode='expand'
                    )

    plt.tight_layout(rect=[0, 0, 1, 1])

    # do this after calling tight layout or changing axes positions in any way:
    fontsize = fig.canvas.get_renderer().points_to_pixels(leg._fontsize)
    pad = 2 * (leg.borderaxespad + leg.borderpad) * fontsize
    leg._legend_box.set_height(leg.get_bbox_to_anchor().height - pad)
    fig.subplots_adjust(left=0.064,bottom=0.09,right=0.62,top=0.922,hspace=0.2,wspace=0.2)
    plt.savefig(test_file_path, format='png',
                dpi=1800)
    plt.show()

def get_plots_custom(plot_path,filename):
    file_data = np.loadtxt(filename, dtype=str)
    legend, blocklens, coderates, mod_types = [], [], [], []
    semlogy={}
    test_file_path = os.path.join(plot_path, 'Noise_SNR_BER')

    for test_row in file_data:
        blocklen, coderate_k, coderate_n, coderate, mod_type, filename, testfilename = test_row
        blocklens.append(blocklen)
        mod_types.append(mod_type)
        coderate_rounded = round(float(coderate), 2)
        coderates.append('{}/{}'.format(coderate_k, coderate_n))

        # data = np.loadtxt(filename)
        test_data = np.loadtxt(testfilename).T
        key_snrs = '{}_{}/{}_{}_{}'.format(blocklen, coderate_k, coderate_n, mod_type,'SNRS')
        key_ray_lc = '{}_{}/{}_{}_{}'.format(blocklen, coderate_k, coderate_n, mod_type,'LC_RAYLEIGH')
        key_awgn_lc = '{}_{}/{}_{}_{}'.format(blocklen, coderate_k, coderate_n, mod_type,'LC_AWGN')
        key_rician_lc = '{}_{}/{}_{}_{}'.format(blocklen, coderate_k, coderate_n, mod_type,'LC_RICIAN')
        key_ray = '{}_{}/{}_{}_{}'.format(blocklen, coderate_k, coderate_n, mod_type, 'RAYLEIGH')
        key_awgn = '{}_{}/{}_{}_{}'.format(blocklen, coderate_k, coderate_n, mod_type, 'AWGN')
        key_rician = '{}_{}/{}_{}_{}'.format(blocklen, coderate_k, coderate_n, mod_type, 'RICIAN')
        semlogy[key_snrs] = test_data[3, :]
        semlogy[key_awgn_lc] = test_data[4, :]
        semlogy[key_ray_lc] = test_data[5, :]
        semlogy[key_rician_lc] = test_data[6, :]
        semlogy[key_awgn] = test_data[7, :]
        semlogy[key_ray] = test_data[8, :]
        semlogy[key_rician] = test_data[9, :]


    for blocklen in np.unique(blocklens):
        for coderate in np.unique(coderates):
            fig = plt.figure()
            ax = plt.subplot(111)
            legend=[]
            for mod_type in np.unique(mod_types):
                key_snrs = '{}_{}_{}_{}'.format(blocklen, coderate, mod_type, 'SNRS')
                key_awgn_lc = '{}_{}_{}_{}'.format(blocklen, coderate, mod_type, 'LC_AWGN')
                key_ray_lc = '{}_{}_{}_{}'.format(blocklen, coderate, mod_type, 'LC_RAYLEIGH')
                key_rician_lc = '{}_{}_{}_{}'.format(blocklen, coderate, mod_type, 'LC_RICIAN')
                key_awgn = '{}_{}_{}_{}'.format(blocklen, coderate, mod_type, 'AWGN')
                key_ray = '{}_{}_{}_{}'.format(blocklen, coderate, mod_type, 'RAYLEIGH')
                key_rician = '{}_{}_{}_{}'.format(blocklen, coderate, mod_type, 'RICIAN')
                snrs=semlogy[key_snrs]
                LC_awgn_bers = semlogy[key_awgn_lc]
                LC_ray_bers = semlogy[key_ray_lc]
                LC_rici_bers = semlogy[key_rician_lc]
                awgn_bers=semlogy[key_awgn]
                ray_bers=semlogy[key_ray]
                rici_bers=semlogy[key_rician]
                ax.semilogy(snrs, LC_awgn_bers, '--', snrs, LC_ray_bers, 'x-', snrs, LC_rici_bers, 'o-',
                            snrs, awgn_bers, '<-', snrs, ray_bers, '>-', snrs, rici_bers, '|-')
                legend.append('LC_AWGN_{}_{}_{}'.format(blocklen, coderate, mod_type))
                legend.append('LC_Rayleigh_{}_{}_{}'.format(blocklen, coderate, mod_type))
                legend.append('LC_Rician_{}_{}_{}'.format(blocklen, coderate, mod_type))
                legend.append('AWGN_{}_{}_{}'.format(blocklen, coderate, mod_type))
                legend.append('Rayleigh_{}_{}_{}'.format(blocklen, coderate, mod_type))
                legend.append('Rician_{}_{}_{}'.format(blocklen, coderate, mod_type))

            plt.grid()
            plt.xticks(snrs)
            plt.title('With blocklen - {}/ coderates {}/ modulation types {} snr-ber'.format(blocklen,
                                                                                             coderate,
                                                                                             np.unique(mod_types)),loc='left')
            legend.sort()
            leg = ax.legend(legend, bbox_to_anchor=(1.04, 0.0, 0.2, 1), loc="upper left", ncol=1, fontsize='small'
                            # ,borderaxespad=0, mode='expand'
                            )

            plt.tight_layout(rect=[0, 0, 1, 1])

            # do this after calling tight layout or changing axes positions in any way:
            fontsize = fig.canvas.get_renderer().points_to_pixels(leg._fontsize)
            pad = 2 * (leg.borderaxespad + leg.borderpad) * fontsize
            leg._legend_box.set_height(leg.get_bbox_to_anchor().height - pad)
            fig.subplots_adjust(left=0.064, bottom=0.09, right=0.8, top=0.922, hspace=0.2, wspace=0.2)
            codes = coderate.split('/')
            coderatek=codes[0]
            coderaten = codes[1]
            testfilename = 'Noise_SNR_BER_{}_{}_{}.png'.format(blocklen,coderatek,coderaten)
            plt.savefig(os.path.join(plot_path,testfilename), format='png', bbox_inches='tight',
                        dpi=1200)
            plt.show()
def get_plots(plot_path,filename):
    get_plots_all(plot_path,filename)
    get_plots_custom(plot_path,filename)