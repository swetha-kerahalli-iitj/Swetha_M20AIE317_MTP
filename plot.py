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
<<<<<<< Updated upstream
=======
from scipy import special

from utils import get_theo_ber, get_modem
>>>>>>> Stashed changes

rcParams['legend.numpoints'] = 1
mpl.style.use('seaborn')
params = {
    'font.family': 'sans-serif',
    'font.sans-serif': 'Times New Roman',
    'font.weight': 'bold'
}
pylab.rcParams.update(params)

def plot(lr):
    fig, ax = plt.subplots(1, 3, figsize=(15, 6), tight_layout=True)
    ax = ax.ravel()
    s = ['Loss', 'BER', 'BLER']
    snrs =  [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    bler_d1_1_001 = [0.9964999999999999, 0.9902999999999998, 0.9718, 0.9379, 0.8751, 0.7842, 0.6458999999999999, 0.5028, 0.35530000000000006, 0.23699999999999996, 0.1437, 0.07880000000000002]
    bler_d10_1_001 = [0.9942, 0.9829000000000001, 0.9583, 0.9091999999999999, 0.8275, 0.7047999999999999, 0.5689, 0.4257000000000001, 0.2861, 0.185, 0.10680000000000003, 0.05400000000000003]
    bler_d1_5_0001 = [0.9963799999999998, 0.9896399999999999, 0.9720199999999997, 0.9386000000000003, 0.8721799999999998, 0.7729800000000003, 0.6465600000000002, 0.50378, 0.35778, 0.24078000000000008, 0.14485999999999996, 0.08214000000000002]
    bler_d10_5_0001 = [0.9920999999999998, 0.98068, 0.9511600000000004, 0.8996600000000002, 0.8137599999999998, 0.69864, 0.5549600000000001, 0.4095600000000001, 0.2833400000000001, 0.17533999999999994, 0.10410000000000003, 0.05456]

    d1_1_001 = np.loadtxt("./data/data_awgn_lr_0.01_D1_10000.txt",usecols=[1, 2], ndmin=2).T
    d10_1_001 = np.loadtxt("./data/data_awgn_lr_0.01_D10_10000.txt",usecols=[1, 2], ndmin=2).T
    d1_5_0001 = np.loadtxt("./data/data_awgn_lr_0.001_D1_50000.txt",usecols=[1, 2], ndmin=2).T
    d10_5_0001 = np.loadtxt("./data/data_awgn_lr_0.001_D10_50000.txt",usecols=[1, 2], ndmin=2).T
    n = d1_1_001.shape[1]  # number of rows
    x = range(0,n)
    for i in range(2):
        if lr==0.01:
            y1 = d1_1_001[i, x]
            y2 = d10_1_001[i, x]
        else:
            y1 = d1_5_0001[i, x]
            y2 = d10_5_0001[i, x]
        ax[i].plot(x, y1, '-', c='#e41b1b', label="D=1", linewidth=2, markersize=6)
        ax[i].plot(x, y2, '-', c='#377eb8', label="D=10", linewidth=2, markersize=6)
        ax[i].set_title(s[i],fontweight='bold',fontsize=16)
        ax[i].legend(loc='best',fancybox=True, framealpha=0,fontsize=14)
        ax[i].grid(True, linestyle='dotted')  # X

    if lr==0.01:
        ax[2].plot(snrs, bler_d1_1_001, '-', c='#e41b1b', label="D=1", linewidth=2, markersize=6)
        ax[2].plot(snrs, bler_d10_1_001, '-', c='#377eb8', label="D=10", linewidth=2, markersize=6)
        ax[2].set_title(s[2],fontweight='bold',fontsize=16)
        ax[2].legend(loc='best',fancybox=True, framealpha=0,fontsize=14)
        ax[2].grid(True, linestyle='dotted')
    else:
        ax[2].plot(snrs, bler_d1_5_0001, '-', c='#e41b1b', label="D=1", linewidth=2, markersize=6)
        ax[2].plot(snrs, bler_d10_5_0001, '-', c='#377eb8', label="D=10", linewidth=2, markersize=6)
        ax[2].set_title(s[2],fontweight='bold',fontsize=16)
        ax[2].legend(loc='best',fancybox=True, framealpha=0,fontsize=14)
        ax[2].grid(True, linestyle='dotted')
    plt.savefig('lr'+str(lr)+'.png', format='png', bbox_inches='tight', transparent=True, dpi=800)
    plt.show()

<<<<<<< Updated upstream
def plot_attn(lr,D,filenamed1,filenamed10):
    fig, ax = plt.subplots(1, 3, figsize=(15, 6), tight_layout=True)
=======

def plot_attn(lr, D, filename, legend):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6), tight_layout=True)
>>>>>>> Stashed changes
    ax = ax.ravel()
    s = ['Loss', 'BER', 'BLER']
    filename1 = './data/data_awgn_lr_'+str(lr)+'_D1_10000.txt'
    filename2 = './data/data_awgn_lr_'+str(lr)+'_D10_10000.txt'

    data1 = np.loadtxt(filename1,usecols=[1, 2], ndmin=2).T
    data2 = np.loadtxt(filename2,usecols=[1, 2], ndmin=2).T
    attn_data1 = np.loadtxt(filenamed1,usecols=[1, 2], ndmin=2).T
    attn_data2 = np.loadtxt(filenamed10,usecols=[1, 2], ndmin=2).T

    snrs =  [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    bler1 = [0.9964999999999999, 0.9902999999999998, 0.9718, 0.9379, 0.8751, 0.7842, 0.6458999999999999, 0.5028, 0.35530000000000006, 0.23699999999999996, 0.1437, 0.07880000000000002]
    bler2 = [0.9942, 0.9829000000000001, 0.9583, 0.9091999999999999, 0.8275, 0.7047999999999999, 0.5689, 0.4257000000000001, 0.2861, 0.185, 0.10680000000000003, 0.05400000000000003]
    attn_bler1 = [0.9963, 0.9907, 0.976, 0.9396000000000001, 0.8700999999999999, 0.7794999999999999, 0.6569999999999999, 0.5147, 0.36810000000000004, 0.238, 0.1451, 0.08390000000000003]
    attn_bler2 = [0.9799000000000001, 0.9519999999999997, 0.8960000000000001, 0.7959000000000002, 0.6552, 0.5089999999999999, 0.36280000000000007, 0.24710000000000001, 0.1608, 0.09130000000000002, 0.05030000000000001, 0.029100000000000008]

    n = data1.shape[1]  # number of rows
    x = range(0,n)
    for i in range(2):
        if D==1:
            y1 = data1[i, x]
<<<<<<< Updated upstream
            y2 = attn_data1[i, x]
        if D==10:
            y1 = data2[i, x]
            y2 = attn_data2[i, x]
        ax[i].plot(x, y1, '--', c='#e41b1b', label="no attention D={}".format(D), linewidth=2, markersize=6)
        ax[i].plot(x, y2, '-', c='#377eb8', label="with attention D={}".format(D), linewidth=2, markersize=6)
        ax[i].set_title(s[i],fontweight='bold',fontsize=16)
        ax[i].legend(loc='best',fancybox=True, framealpha=0,fontsize=14)
        ax[i].grid(True, linestyle='dotted')  # X
=======
        ax[i].plot(x, y1, 'o--', c='#e41b1b', linewidth=2, markersize=6)
        ax[i].set_title(s[i], fontweight='bold', fontsize=16)
        ax[i].set_ylabel(s[i], fontweight='bold', fontsize=16)
        ax[i].set_xlabel('Epoch', fontweight='bold', fontsize=16)
        ax[i].set_xticks(x)
        if i > 0:
            ax[i].set_yscale('logit')

        ax[i].legend(loc='best', fancybox=True, framealpha=0, fontsize=14)
        ax[i].grid(True, linestyle='dotted')  # X

>>>>>>> Stashed changes

    if D==1:
        ax[2].plot(snrs, bler1, '--', c='#e41b1b', label="no attention D={}".format(D), linewidth=2, markersize=6)
        ax[2].plot(snrs, attn_bler1, '-', c='#377eb8', label="with attention D={}".format(D), linewidth=2, markersize=6)
    if D==10:
        ax[2].plot(snrs, bler2, '--', c='#e41b1b', label="no attention D={}".format(D), linewidth=2, markersize=6)
        ax[2].plot(snrs, attn_bler2, '-', c='#377eb8', label="with attention D={}".format(D), linewidth=2, markersize=6)
    ax[2].set_title(s[2],fontweight='bold',fontsize=16)
    ax[2].legend(loc='best',fancybox=True, framealpha=0,fontsize=14)
    ax[2].grid(True, linestyle='dotted')
    plt.savefig('./data_test/plots/attention_D'+str(D)+'.png', format='png', bbox_inches='tight', transparent=True, dpi=800)
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
<<<<<<< Updated upstream
    'ylabel': 'BER',
    'savepath': './data/plots/awgn_lr0.01_D10_50000_snr.png',
    'fig_size': (9,6),
=======
    'ylabel': 'BLER',
    'savepath': 'rayleigh_lr0.01_D10_50000_snr.png',
    'fig_size': (9, 6),
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
def plot_snr(filename,legend):
=======


def plot_snr_test(filename, plot_path,timestamp):
    test_data = np.loadtxt(filename).T
    snrs = test_data[0, :]
    ber = test_data[1, :]
    awgn = test_data[2, :]
    rayleigh =test_data[3, :]
    bler = test_data[4, :]
    mod = get_modem()
    mean_BER_theoretical = np.zeros(len(snrs))
    test_file_path = os.path.join(plot_path, 'rayleigh_test_both'+str(timestamp)+'.png')
    test_file_path_act = os.path.join(plot_path, 'rayleigh_test_act_'+str(timestamp)+'.png')
    test_file_path_awgn = os.path.join(plot_path, 'rayleigh_test_awgn_'+str(timestamp)+'.png')
    test_file_path_ray = os.path.join(plot_path, 'rayleigh_test_ray_' + str(timestamp) + '.png')
    test_file_path_exp = os.path.join(plot_path, 'rayleigh_test_exp_' + str(timestamp) + '.png')
    i = 0
    for SNR in snrs:
        # calculate theoretical BER for BPSK
        mean_BER_theoretical[i] = get_theo_ber(SNR)
        i += 1

    plt.semilogy(snrs, ber, 'o-',snrs,awgn,'o-',snrs,rayleigh,'o-',snrs,mod.calcTheoreticalBER(snrs))
    plt.grid()
    plt.xlabel('Signal to Noise Ration (dB)')
    plt.ylabel('Bit Error Rate')
    plt.xticks(snrs)
    # plt.yticks((0.1,0.01))
    plt.title('test-awgn-simulated rayleigh SNR-BER')
    plt.legend(('test rayleigh snr-ber','awgn snr-ber','sim rayleigh snr-ber','theo snr-ber'))
    plt.savefig(test_file_path, format='png', bbox_inches='tight',
                 dpi=1200)
    plt.show()

    fig, ax = plt.subplots(figsize=(12,8))
    ax.semilogy(snrs, mod.calcTheoreticalBER(snrs), "--", label="Theoretical")
    ax.semilogy(snrs, awgn,
                label="AWGN")
    ax.semilogy(snrs, ber, "o-", linewidth=3, markersize=8,
                label="Rayleigh with ML")

    ax.semilogy(snrs, rayleigh,"o-",
                label="Rayleigh")

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
    plt.title('test rayleigh SNR-BER')
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


def plot_snr(filename, legend):
>>>>>>> Stashed changes
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

def plot(filename, legend):
    data = np.loadtxt(filename)
    X = data[:,0]
    Y = data[:,2]
    plt.figure(figsize=legend['fig_size'])
    plt.title(legend['title'], fontsize=legend['title_size'])
    ax = plt.subplot(111)
    plt.xticks(fontsize=legend['label_fontsize']-2)  # x cordinate
    plt.yticks(fontsize=legend['label_fontsize']-2)  # Y cordinate
    l1 = ax.plot(X, Y, color=colors[0], linestyle=line_styles[0], linewidth=legend['linewidth'], label=legend['ylabel'])
    ax.set_xlabel(legend['xlabel'], fontsize=legend['label_fontsize'])
    ax.set_ylabel(legend['ylabel'], fontsize=legend['label_fontsize'])
    plt.savefig(legend['savepath'])
    plt.show()

def get_plots(filenames,lr):
    d1FilenamesList = glob.glob('./data_test/data_awgn_lr_'+str(lr)+'_D1_*.txt')
    d10FilenamesList = glob.glob('./data_test/data_awgn_lr_'+str(lr)+'_D10_*.txt')
    d1FilenamesList[:0]=d1FilenamesList
    d1FilenamesList[:0] = d1FilenamesList
    # plot_attn(lr, 1, filename)

    for filename in d1FilenamesList:
        plot(filename, legend)
        plot_snr(filename, legend_snr)

