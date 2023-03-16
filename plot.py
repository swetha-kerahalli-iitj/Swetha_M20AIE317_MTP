'''
Input file follows the following format. All lines are
mandatory. No comments can be made within the file.
Must be followed strictly.
'''
import glob
import math

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os, sys
import numpy as np

from pylab import rcParams
import matplotlib.pylab as pylab

from utils import get_modem


def plot_attn(lr, D, filename, legend):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6), tight_layout=True)

    ax = ax.ravel()
    s = ['Loss', 'BER']

    data1 = np.loadtxt(filename, usecols=[1, 2], ndmin=2).T

    n = data1.shape[1]  # number of rows
    x = range(0, n)
    for i in range(2):
        if D==1:
            y1 = data1[i, x]

        ax[i].plot(x, y1, 'o--', c='#e41b1b', linewidth=2, markersize=6)
        ax[i].set_title(s[i], fontweight='bold', fontsize=16)
        ax[i].set_ylabel(s[i], fontweight='bold', fontsize=16)
        ax[i].set_xlabel('Epoch', fontweight='bold', fontsize=16)
        ax[i].set_xticks(x)
        if i > 0:
            ax[i].set_yscale('logit')

        ax[i].legend(loc='best', fancybox=True, framealpha=0, fontsize=14)
        ax[i].grid(True, linestyle='dotted')  # X
    plt.savefig(legend['savepath'], format='png', bbox_inches='tight', transparent=True,
                dpi=800)
    plt.show()

colors = ['#C95F63', '#F1AD32', '#3B8320', '#516EA9',  '#292DF4', ]
line_styles = ['-']  # ['-', ':',  '--','-.']
marker_types = ['o', 's', 'v', '^', '*', 'h']

# This lists out all the variables that you can control
# A copy of this dictionary will be generated (deepcopy),
# in case the default values are lost
legend = {
    'title': 'Delay=10 num_block=50000 lr=0.01',
    'xlabel': 'Epoch',
    'ylabel': 'BER',
    'savepath': './data/plots/awgn_lr0.01_D10_50000.png',
    'fig_size': (9,6),
    'label_fontsize': 15,
    'markersize': 10,
    'linewidth': 2,
    'title_size': 20,
    'loc': 5,  # location of legend, see online documentation
    'plot_type': 'line',  # line = line, scatter = scatter, both = line + intervaled marker
    'x_range': 'auto',  # auto, 0 100
    'y_range': 'auto',
    'line_length': 40,
    'markevery': 0.5
}

legend_snr = {
    'title': 'Delay=10 num_block=50000 lr=0.01',
    'xlabel': 'SNR',
    'ylabel': 'BLER',
    'savepath': 'rayleigh_lr0.01_D10_50000_snr.png',
    'fig_size': (9, 6),
    'label_fontsize': 15,
    'markersize': 10,
    'linewidth': 2,
    'title_size': 20,
    'loc': 5,  # location of legend, see online documentation
    'plot_type': 'line',  # line = line, scatter = scatter, both = line + intervaled marker
    'x_range': 'auto',  # auto, 0 100
    'y_range': 'auto',
    'line_length': 40,
    'markevery': 0.5
}

def plot_snr_test(args,filename, timestamp):
    test_data = np.loadtxt(filename).T
    snrs = test_data[0, :]
    ber = test_data[1, :]
    awgn = test_data[2, :]
    rayleigh =test_data[3, :]
    rician = test_data[4, :]
    bler = test_data[5, :]
    mod = get_modem()
    plot_path = args.PLOT_PATH
    test_file_path = os.path.join(plot_path, 'rayleigh_test_both'+str(timestamp)+'.png')
    test_file_path_act = os.path.join(plot_path, 'rayleigh_test_act_'+str(timestamp)+'.png')
    test_file_path_awgn = os.path.join(plot_path, 'rayleigh_test_awgn_'+str(timestamp)+'.png')
    test_file_path_ray = os.path.join(plot_path, 'rayleigh_test_ray_' + str(timestamp) + '.png')
    test_file_path_exp = os.path.join(plot_path, 'rayleigh_test_exp_' + str(timestamp) + '.png')

    plt.semilogy(snrs, ber, 'o-',snrs,awgn,'o-',snrs,rayleigh,'o-',snrs,rician,'o-',snrs,mod.calcTheoreticalBER(snrs))
    plt.grid()
    plt.xlabel('Signal to Noise Ration (dB)')
    plt.ylabel('Bit Error Rate')
    plt.xticks(snrs)
    # plt.yticks((0.1,0.01))
    plt.title('test-simulated {} SNR-BER'.format(args.Simulate))
    plt.legend(('test {} snr-ber'.format(args.Simulate),'AWGN snr-ber','sim Rayleigh snr-ber','sim Rician snr-ber','theo snr-ber'))
    plt.savefig(test_file_path, format='png', bbox_inches='tight',
                 dpi=1200)
    plt.show()

    fig, ax = plt.subplots(figsize=(12,8))
    ax.semilogy(snrs, mod.calcTheoreticalBER(snrs), "--", label="Theoretical")
    ax.semilogy(snrs, awgn,
                label="AWGN")
    ax.semilogy(snrs, ber, "o-", linewidth=3, markersize=8,
                label="{} with ML".format(args.Simulate))

    ax.semilogy(snrs, rayleigh,"o-",
                label="Rayleigh")
    ax.semilogy(snrs, rician, "o-",
                label="Rician")

    ax.set_title("16-QAM Error Probability")
    ax.set_ylabel("Bit Error Rate (BER)")
    new_list = range(math.floor(min(snrs)), math.ceil(max(snrs)) + 1)
    ax.set_xticks(new_list)
    ax.set_xlabel("SNR (dB)")
    ax.legend()
    plt.savefig(test_file_path_exp, format='png', bbox_inches='tight',
                dpi=1200)
    plt.show()

    plt.semilogy(snrs, ber, 'o-')
    plt.grid()
    plt.xticks(snrs)
    # plt.yticks((0,0.1, 0.01))
    plt.xlabel('Signal to Noise Ration (dB)')
    plt.ylabel('Bit Error Rate')
    plt.title('test {} SNR-BER'.format(args.Simulate))
    plt.legend('test snr-ber')
    plt.savefig(test_file_path_act, format='png', bbox_inches='tight',
                 dpi=1200)
    plt.show()

    plt.semilogy( snrs, awgn, 'o-')
    plt.grid()
    plt.xticks(snrs)
    # plt.yticks((0,0.1, 0.01))
    plt.xlabel('Signal to Noise Ration (dB)')
    plt.ylabel('Bit Error Rate')
    plt.title('awgn SNR-BER')
    plt.legend( 'awgn snr-ber')
    plt.savefig(test_file_path_awgn, format='png', bbox_inches='tight',
                dpi=1200)
    plt.show()

    plt.plot(snrs, rayleigh, 'o-')
    plt.grid()
    plt.yscale('symlog')
    plt.xticks(snrs)
    # plt.yticks((0,0.1, 0.01))
    plt.xlabel('Signal to Noise Ration (dB)')
    plt.ylabel('Bit Error Rate')
    plt.title('simulated rayleigh SNR-BER')
    plt.legend('sim rayleigh snr-ber')
    plt.savefig(test_file_path_ray, format='png', bbox_inches='tight',
               dpi=1200)
    plt.show()

    plt.plot(snrs, rician, 'o-')
    plt.grid()
    plt.yscale('symlog')
    plt.xticks(snrs)
    # plt.yticks((0,0.1, 0.01))
    plt.xlabel('Signal to Noise Ration (dB)')
    plt.ylabel('Bit Error Rate')
    plt.title('simulated rician SNR-BER')
    plt.legend('sim rician snr-ber')
    plt.savefig(test_file_path_ray, format='png', bbox_inches='tight',
                dpi=1200)
    plt.show()


def plot_snr(filename, legend):
    data = np.loadtxt(filename)
    X = data[:, 0]
    Y = data[:, 3]
    # X = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    # Y = [0.0701569989323616, 0.05043799802660942, 0.0348035953938961, 0.022770600393414497, 0.014346200972795486, 0.008601599372923374, 0.004972000140696764, 0.0027095996774733067, 0.0014349999837577343, 0.0007247999892570078, 0.00036559993168339133, 0.00017820001812651753]
    plt.figure(figsize=legend['fig_size'])
    plt.title(legend['title'], fontsize=legend['title_size'])
    ax = plt.subplot(111)
    plt.xticks(fontsize=legend['label_fontsize']-2)  # x cordinate
    plt.yticks(fontsize=legend['label_fontsize']-2)  # Y cordinate
    l1 = ax.plot(X, Y, color=colors[0], linestyle=line_styles[0], marker=marker_types[0], markersize=legend['markersize'], linewidth=legend['linewidth'], label=legend['ylabel'])
    ax.set_xlabel(legend['xlabel'], fontsize=legend['label_fontsize'])
    ax.set_ylabel(legend['ylabel'], fontsize=legend['label_fontsize'])
    plt.savefig(legend['savepath'])
    plt.show()

def get_plots(args, filename='None', test_file_name='None',timestamp='00'):
    legend_snr['savepath'] = os.path.join(args.PLOT_PATH, 'rayleigh_snr_bler_'+timestamp+'.png')
    legend['savepath'] = os.path.join(args.PLOT_PATH, 'rayleigh_epoch_'+timestamp+'.png')

    if filename != 'None':
        # plot(filename, legend)
        plot_attn(0.01, 1, filename, legend)
        plot_snr(filename, legend_snr)
    if test_file_name != 'None':
        plot_snr_test(args,test_file_name, timestamp)

