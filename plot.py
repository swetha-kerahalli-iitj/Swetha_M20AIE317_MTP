import os

import numpy
import numpy as np
from matplotlib import pyplot as plt

def get_plots_all(plot_path,filename):
    file_data = np.loadtxt(filename, dtype=str)
    semlogy, legend, blocklens, coderates, channels = [], [], [], [], []

    test_file_path = os.path.join(plot_path, 'All_Noise_SNR_BER' + '.png')

    fig = plt.figure()
    ax = plt.subplot(111)
    for test_row in file_data:
        blocklen, coderate_k, coderate_n, coderate, channel, mod_type,filename, testfilename = test_row
        blocklens.append(blocklen)
        channels.append(channel)
        coderate_rounded = round(float(coderate), 2)
        coderates.append('{}/{}'.format(coderate_k, coderate_n))

        # data = np.loadtxt(filename)
        test_data = np.loadtxt(testfilename).T
        snrs = test_data[3, :]
        test_ber = test_data[4, :]
        punc_ber = test_data[5, :]
        legend.append('LC_{}_{}/{}_{}'.format(blocklen, coderate_k, coderate_n, channel))
        # legend.append('LC_Rayleigh_{}_{}/{}_{}'.format(blocklen, coderate_k, coderate_n, channels))
        # legend.append('LC_Rician_{}_{}/{}_{}'.format(blocklen, coderate_k, coderate_n, channels))
        # legend.append('SIM_AWGN_{}_{}/{}_{}'.format(blocklen, coderate_k, coderate_n, channels))
        # legend.append('SIM_Rayleigh_{}_{}/{}_{}'.format(blocklen, coderate_k, coderate_n, channels))
        # legend.append('SIM_Rician_{}_{}/{}_{}'.format(blocklen, coderate_k, coderate_n, channels))
        ax.semilogy(snrs, test_ber, '--')

    plt.grid()
    plt.xticks(snrs)
    plt.title('With blocklen - {}/ coderates {}/ channel types {} snr-ber'.format(np.unique(blocklens),
                                                                                     np.unique(coderates),
                                                                                     np.unique(channels)),loc='left')

    leg = ax.legend(legend,bbox_to_anchor=(1.04, 0.0, 0.3, 1), loc="upper left",ncol=2,fontsize='xx-small'
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
    legend, blocklens, coderates, channels = [], [], [], []
    semlogy={}
    test_file_path = os.path.join(plot_path, 'Noise_SNR_BER')
    color_legend= ("#CBF296","#BDF296","#AEF296","#A3F296","#97F296","#96F2A4","#96F2B2","#96F2BA","#96F2CB")
    color_legend_LC =( "#D296F2","#DC96F2","#EB96F2","#F296F1","#F296EE","#F296E3","#F296D8","#F296D2","#F296C5")
    for test_row in file_data:
        blocklen, coderate_k, coderate_n, coderate, channel, mod_type,filename, testfilename = test_row
        blocklens.append(blocklen)
        channels.append(channel)
        coderate_rounded = round(float(coderate), 2)
        coderates.append('{}/{}'.format(coderate_k, coderate_n))

        # data = np.loadtxt(filename)
        test_data = np.loadtxt(testfilename).T
        key_snrs = '{}_{}/{}_{}_{}'.format(blocklen, coderate_k, coderate_n, channel,'SNRS')
        key_lc = '{}_{}/{}_{}_{}'.format(blocklen, coderate_k, coderate_n, channel,'LC')
        # key_awgn_lc = '{}_{}/{}_{}_{}'.format(blocklen, coderate_k, coderate_n, channel,'LC_AWGN')
        # key_rician_lc = '{}_{}/{}_{}_{}'.format(blocklen, coderate_k, coderate_n, channel,'LC_RICIAN')
        # key_ray = '{}_{}/{}_{}_{}'.format(blocklen, coderate_k, coderate_n, channel, 'RAYLEIGH')
        # key_awgn = '{}_{}/{}_{}_{}'.format(blocklen, coderate_k, coderate_n, channel, 'AWGN')
        # key_rician = '{}_{}/{}_{}_{}'.format(blocklen, coderate_k, coderate_n, channel, 'RICIAN')
        semlogy[key_snrs] = test_data[3, :]
        semlogy[key_lc] = test_data[4, :]
        # semlogy[key_ray_lc] = test_data[5, :]
        # semlogy[key_rician_lc] = test_data[6, :]
        # semlogy[key_awgn] = test_data[7, :]
        # semlogy[key_ray] = test_data[8, :]
        # semlogy[key_rician] = test_data[9, :]


    for blocklen in np.unique(blocklens):
        for coderate in np.unique(coderates):
            fig = plt.figure()
            ax = plt.subplot(111)
            legend=[]
            i=0
            marker=0
            for channel in np.unique(channels):
                key_snrs = '{}_{}_{}_{}'.format(blocklen, coderate, channel, 'SNRS')
                key_lc = '{}_{}_{}_{}'.format(blocklen, coderate, channel, 'LC')
                # key_ray_lc = '{}_{}_{}_{}'.format(blocklen, coderate, channel, 'LC_RAYLEIGH')
                # key_rician_lc = '{}_{}_{}_{}'.format(blocklen, coderate, channel, 'LC_RICIAN')
                # key_awgn = '{}_{}_{}_{}'.format(blocklen, coderate, channel, 'AWGN')
                # key_ray = '{}_{}_{}_{}'.format(blocklen, coderate, channel, 'RAYLEIGH')
                # key_rician = '{}_{}_{}_{}'.format(blocklen, coderate, channel, 'RICIAN')
                snrs=semlogy[key_snrs]
                LC_bers = semlogy[key_lc]
                # LC_ray_bers = semlogy[key_ray_lc]
                # LC_rici_bers = semlogy[key_rician_lc]
                # awgn_bers=semlogy[key_awgn]
                # ray_bers=semlogy[key_ray]
                # rici_bers=semlogy[key_rician]
                color_legend_LC = color_legend[i]
                if channel == 'awgn':
                    color_legend_LC = color_legend_LC[i]
                ax.semilogy(snrs, LC_bers,linewidth = 2,marker = 'o',markersize = marker+2)
                # ax.semilogy(snrs, LC_ray_bers,color= color_legend_LC[i+1] ,linewidth = 2,marker = 'o',markersize = marker)
                # ax.semilogy(snrs, LC_rici_bers,color= color_legend_LC[i+2] ,linewidth = 2,marker = 'o',markersize = marker+2)
                # ax.semilogy(snrs, awgn_bers,color= color_legend[i] ,linewidth = 1,marker = 'x',markersize = marker)
                # ax.semilogy(snrs, ray_bers,color= color_legend[i+1] ,linewidth = 1,marker = 'x',markersize = marker)
                # ax.semilogy(snrs, rici_bers,color= color_legend[i+2] ,linewidth = 1,marker = 'x',markersize = marker)
                # i = i+2
                marker += 1
                legend.append('LC_{}_{}_{}'.format(blocklen, coderate, channel))
                # legend.append('LC_Rayleigh_{}_{}_{}'.format(blocklen, coderate, mod_type))
                # legend.append('LC_Rician_{}_{}_{}'.format(blocklen, coderate, mod_type))
                # legend.append('SIM_AWGN_{}_{}_{}'.format(blocklen, coderate, mod_type))
                # legend.append('SIM_Rayleigh_{}_{}_{}'.format(blocklen, coderate, mod_type))
                # legend.append('SIM_Rician_{}_{}_{}'.format(blocklen, coderate, mod_type))

            plt.grid()
            plt.xticks(snrs)
            plt.title('With blocklen - {}/ coderates {}/ channel types {} snr-ber'.format(blocklen,
                                                                                             coderate,
                                                                                             np.unique(channels)),loc='left')
            # legend.sort()
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