'''
Input file follows the following format. All lines are
mandatory. No comments can be made within the file.
Must be followed strictly.
'''
import glob
import os

import matplotlib as mpl
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams

rcParams['legend.numpoints'] = 1
mpl.style.use('seaborn')
params = {
    'font.family': 'sans-serif',
    'font.sans-serif': 'Times New Roman',
    'font.weight': 'bold'
}
pylab.rcParams.update(params)

def plot(lr,D):
    fig, ax = plt.subplots(1, 3, figsize=(15, 6), tight_layout=True)
    ax = ax.ravel()
    s = ['Loss', 'BER', 'BLER']
    # snrs =  [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    # bler_d1_1_001 = [0.9964999999999999, 0.9902999999999998, 0.9718, 0.9379, 0.8751, 0.7842, 0.6458999999999999, 0.5028, 0.35530000000000006, 0.23699999999999996, 0.1437, 0.07880000000000002]
    # bler_d10_1_001 = [0.9942, 0.9829000000000001, 0.9583, 0.9091999999999999, 0.8275, 0.7047999999999999, 0.5689, 0.4257000000000001, 0.2861, 0.185, 0.10680000000000003, 0.05400000000000003]
    # bler_d1_5_0001 = [0.9963799999999998, 0.9896399999999999, 0.9720199999999997, 0.9386000000000003, 0.8721799999999998, 0.7729800000000003, 0.6465600000000002, 0.50378, 0.35778, 0.24078000000000008, 0.14485999999999996, 0.08214000000000002]
    # bler_d10_5_0001 = [0.9920999999999998, 0.98068, 0.9511600000000004, 0.8996600000000002, 0.8137599999999998, 0.69864, 0.5549600000000001, 0.4095600000000001, 0.2833400000000001, 0.17533999999999994, 0.10410000000000003, 0.05456]

    # snrs = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    snrs = [-1.4999997446509226, -1.0000000166986343, -0.49999973308696327, -0.0, 0.5000001308463472, 1.0000002900227403, 1.5000000201403676, 2.0000002404171053, 2.5000000877622415, 3.0000002493010487, 3.500000207085638, 3.999999717024358]
    ber_d1 = [0.06675098091363907, 0.05408501252532005, 0.04294999688863754, 0.03319299593567848, 0.025297999382019043, 0.018310001119971275, 0.012911999598145485, 0.008712999522686005, 0.0058019994758069515, 0.0035670006182044744, 0.002120000310242176, 0.0012459997087717056]
    bler_d1 = [0.9976999999999998, 0.9926999999999998, 0.9778999999999998, 0.9552999999999996, 0.9033000000000002, 0.8141999999999999, 0.6979999999999997, 0.5603999999999998, 0.42700000000000005, 0.2877999999999999, 0.18660000000000007, 0.11429999999999996]
    ber_d10 =[0.07044699043035507, 0.05925600230693817, 0.0488710030913353, 0.03905600309371948, 0.03105800226330757, 0.02360299602150917, 0.017810998484492302, 0.012830005027353764, 0.009007997810840607, 0.006154997274279594, 0.004013999365270138, 0.002441000659018755]
    bler_d10 =[0.9991999999999999, 0.9975999999999996, 0.9915000000000002, 0.9798999999999991, 0.9525999999999999, 0.9045000000000003, 0.8342999999999998, 0.7151, 0.5929000000000001, 0.46180000000000004, 0.3278999999999999, 0.21520000000000006]

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
        ax[2].plot(snrs, bler_d1, '-', c='#e41b1b', label="D=1", linewidth=2, markersize=6)
        ax[2].plot(snrs, bler_d10, '-', c='#377eb8', label="D=10", linewidth=2, markersize=6)
        ax[2].set_title(s[2],fontweight='bold',fontsize=16)
        ax[2].legend(loc='best',fancybox=True, framealpha=0,fontsize=14)
        ax[2].grid(True, linestyle='dotted')
    # else:
    #     ax[2].plot(snrs, bler_d1_5_0001, '-', c='#e41b1b', label="D=1", linewidth=2, markersize=6)
    #     ax[2].plot(snrs, bler_d10_5_0001, '-', c='#377eb8', label="D=10", linewidth=2, markersize=6)
    #     ax[2].set_title(s[2],fontweight='bold',fontsize=16)
    #     ax[2].legend(loc='best',fancybox=True, framealpha=0,fontsize=14)
    #     ax[2].grid(True, linestyle='dotted')
    plt.savefig('lr'+str(lr)+'.png', format='png', bbox_inches='tight', transparent=True, dpi=800)
    plt.show()

def plot_attn(lr,D,attnFilename,path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 6), tight_layout=True)
    ax = ax.ravel()
    s = ['Loss', 'BER', 'BLER']
    xlabel = ['Epoch', 'Epoch', 'SNR']
    ylabel = s
    filename = './data/data_awgn_lr_'+str(lr)+'_D'+str(D)+'_10000.txt'

    data = np.loadtxt(filename,usecols=[1, 2], ndmin=2).T
    attn_data = np.loadtxt(attnFilename,usecols=[1, 2], ndmin=2).T
    #
    # snrs =  [-1.4999997446509226, -1.0000000166986343, -0.49999973308696327, -0.0, 0.5000001308463472, 1.0000002900227403, 1.5000000201403676, 2.0000002404171053, 2.5000000877622415, 3.0000002493010487, 3.500000207085638, 3.999999717024358]
    bler1 = [ 0.9379, 0.8751, 0.7842, 0.6458999999999999, 0.5028, 0.35530000000000006, 0.23699999999999996, 0.1437, 0.07880000000000002]
    bler2 = [ 0.9091999999999999, 0.8275, 0.7047999999999999, 0.5689, 0.4257000000000001, 0.2861, 0.185, 0.10680000000000003, 0.05400000000000003]
    # attn_bler1 = [0.9976999999999998, 0.9926999999999998, 0.9778999999999998, 0.9552999999999996, 0.9033000000000002, 0.8141999999999999, 0.6979999999999997, 0.5603999999999998, 0.42700000000000005, 0.2877999999999999, 0.18660000000000007, 0.11429999999999996]
    # attn_bler2 = [0.9991999999999999, 0.9975999999999996, 0.9915000000000002, 0.9798999999999991, 0.9525999999999999, 0.9045000000000003, 0.8342999999999998, 0.7151, 0.5929000000000001, 0.46180000000000004, 0.3278999999999999, 0.21520000000000006]
    snrs = [ 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    attn_bler1 = [ 0.9552999999999996, 0.9033000000000002, 0.8141999999999999, 0.6979999999999997, 0.5603999999999998, 0.42700000000000005, 0.2877999999999999, 0.18660000000000007, 0.11429999999999996]
    attn_bler2 = [ 0.9798999999999991, 0.9525999999999999, 0.9045000000000003, 0.8342999999999998, 0.7151, 0.5929000000000001, 0.46180000000000004, 0.3278999999999999, 0.21520000000000006]
    n = attn_data.shape[1]  # number of rows
    x = range(0,n)
    for i in range(2):
        if(i==1):
            y1 = [10*j for j in data[i, x]]
            y2 = [10*j for j in attn_data[i, x]]
        else:
            y1 = data[i, x]
            y2 = attn_data[i, x]
        ax[i].plot(x, y1, '--', c='#e41b1b', label="no attention D={}".format(D), linewidth=2, markersize=6)
        ax[i].plot(x, y2, '-', c='#377eb8', label="with attention D={}".format(D), linewidth=2, markersize=6)
        ax[i].set_xlabel(xlabel[i], fontsize=14)
        ax[i].set_ylabel(ylabel[i], fontsize=14)
        ax[i].set_title(s[i],fontweight='bold',fontsize=16)
        ax[i].legend(loc='best',fancybox=True, framealpha=0,fontsize=14)
        ax[i].grid(True, linestyle='dotted')  # X

    if D==1:
        ax[2].plot(snrs, bler1, '--', c='#e41b1b', label="no attention D={}".format(D), linewidth=2, markersize=8)
        ax[2].plot(snrs, attn_bler1, '-', c='#377eb8', label="with attention D={}".format(D), linewidth=2, markersize=8)
    if D==10:
        ax[2].plot(snrs, bler2, '--', c='#e41b1b', label="no attention D={}".format(D), linewidth=2, markersize=8)
        ax[2].plot(snrs, attn_bler2, '-', c='#377eb8', label="with attention D={}".format(D), linewidth=2, markersize=8)
    ax[2].set_xlabel(xlabel[2], fontsize=14)
    ax[2].set_ylabel(ylabel[2], fontsize=14)
    ax[2].set_title(s[2],fontweight='bold',fontsize=16)
    ax[2].legend(loc='best',fancybox=True, framealpha=0,fontsize=14)
    ax[2].grid(True, linestyle='dotted')
    plt.savefig('./data_new/plots/attention_D'+str(D)+'.png', format='png', bbox_inches='tight', transparent=True, dpi=800)
    plt.show()

colors = ['#C95F63', '#F1AD32', '#3B8320', '#516EA9',  '#292DF4', ]
line_styles = ['-']  # ['-', ':',  '--','-.']
marker_types = ['o', 's', 'v', '^', '*', 'h']

# This lists out all the variables that you can control
# A copy of this dictionary will be generated (deepcopy),
# in case the default values are lost
legend = {
    'title': 'Delay=1 num_block=10000 lr=0.01',
    'xlabel': 'Epoch',
    'ylabel': 'BER',
    'savepath': './data_new/plots/awgn_lr_0.01_D10_10000.png',
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
    'title': 'Delay=1 num_block=10000 lr=0.01',
    'xlabel': 'SNR',
    'ylabel': 'BER',
    'savepath': './data_new/plots/awgn_lr_0.01_D10_10000_snr.png',
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
def plot_snr(filename,legend,D):
    data = np.loadtxt(filename)
    X =  [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    Y = data[:, 3]
    legend['savepath'] = './data_new/plots/awgn_lr_0.01_D' + str(D) + '_10000_snr.png'
    legend['title'] = 'Delay=' + str(D) + ' num_block=10000 lr=0.01'
    if(D==1):
        ber = [ 0.03319299593567848, 0.025297999382019043, 0.018310001119971275, 0.012911999598145485, 0.008712999522686005, 0.0058019994758069515, 0.0035670006182044744, 0.002120000310242176, 0.0012459997087717056]
    else:
        ber=[0.03905600309371948, 0.03105800226330757, 0.02360299602150917, 0.017810998484492302, 0.012830005027353764, 0.009007997810840607, 0.006154997274279594, 0.004013999365270138, 0.002441000659018755]
    # Y=[10**i for i in ber]
    # Y=np.abs(np.log10(ber))
    Y=[10*i for i in ber]

    plt.figure(figsize=legend['fig_size'])
    plt.title(legend['title'], fontsize=legend['title_size'])
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax = plt.subplot(111)
    plt.xticks(fontsize=legend['label_fontsize']-2)  # x cordinate
    plt.yticks(fontsize=legend['label_fontsize']-2)  # Y cordinate
    # plt.yscale("symlog")

    # plt.yscale("log")

    l1 = ax.plot(X, Y, color=colors[0], linestyle=line_styles[0], marker=marker_types[0], markersize=legend['markersize'], linewidth=legend['linewidth'], label=legend['ylabel'])
    ax.set_xlabel(legend['xlabel'], fontsize=legend['label_fontsize'])
    ax.set_ylabel(legend['ylabel'], fontsize=legend['label_fontsize'])
    plt.savefig(legend['savepath'])
    plt.show()

def plot(filename, legend,D):
    data = np.loadtxt(filename)
    X = data[:,0]
    # Y = data[:,2]
    Y = [10*i for i in data[:,2]]
    legend['savepath'] = './data_new/plots/awgn_lr_0.01_D' + str(D) + '_10000.png'
    legend['title']='Delay='+str(D)+' num_block=10000 lr=0.01'
    plt.figure(figsize=legend['fig_size'])
    plt.title(legend['title'], fontsize=legend['title_size'])
    ax = plt.subplot(111)
    plt.xticks(fontsize=legend['label_fontsize']-2)  # x cordinate
    plt.yticks(fontsize=legend['label_fontsize']-2)  # Y cordinate
    # plt.yscale("log")
    l1 = ax.plot(X, Y, color=colors[0], linestyle=line_styles[0], linewidth=legend['linewidth'], label=legend['ylabel'])
    ax.set_xlabel(legend['xlabel'], fontsize=legend['label_fontsize'])
    ax.set_ylabel(legend['ylabel'], fontsize=legend['label_fontsize'])

    plt.savefig(legend['savepath'])
    plt.show()

def get_plots(lr,path):
    d1FilenamesList = glob.glob(os.path.join(path,r'attention_data_awgn_lr_'+str(lr)+'_D1_*.txt'))
    d10FilenamesList = glob.glob(os.path.join(path,r'attention_data_awgn_lr_'+str(lr)+'_D10_*.txt'))
    # d1FilenamesList[:0]=d1FilenamesList
    # d1FilenamesList[:0] = d1FilenamesList
    # plot_attn(lr, 1, filename)

    for filename in d1FilenamesList:
        plot_attn(lr, 1, filename)
        # plot(filename, legend,1)
        # plot_snr(filename, legend_snr,1)
    for filename in d10FilenamesList:
        plot_attn(lr, 10, filename)
        # plot(filename, legend,10)
        # plot_snr(filename, legend_snr,10)
