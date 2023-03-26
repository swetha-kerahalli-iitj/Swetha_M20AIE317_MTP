import torch
import sys, os
import time
from get_args import get_args
from plot import get_plots
from trainer import train_model


attnFilename=[]
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


if __name__ == '__main__':
    #################################################
    # load args & setup logger
    #################################################
    start_time = time.time()

    args = get_args()
    BASE_PATH = args.BASE_PATH
    LOG_PATH = args.LOG_PATH
    DATA_PATH = args.DATA_PATH
    MODEL_PATH = args.MODEL_PATH
    PLOT_PATH = args.PLOT_PATH
    for path in (LOG_PATH, DATA_PATH, MODEL_PATH, PLOT_PATH):
        if not os.path.isdir(path):
            os.makedirs(path)
    # put all printed things to log file
    if args.init_nw_weight == 'default':
        start_epoch = 1
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    else:
        start_epoch = int(args.init_nw_weight.split('_')[2]) + 1
        timestamp = args.init_nw_weight.split('_')[8].split('.')[0]
    formatfile = 'attention_log_' + str(args.channel) + '_lr_' + str(args.enc_lr) + '_D' + str(args.D) + '_' + str(
        args.num_block) + '_' + timestamp + '.txt'
    logfilename = os.path.join(LOG_PATH, formatfile)
    logfile = open(logfilename, 'a')
    sys.stdout = Logger(logfilename, sys.stdout)

    print(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("use_cuda: ", use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    filename_all = os.path.join(DATA_PATH,
                            'attention_data_' + str(args.channel) + '_lr_' + str(args.enc_lr) + '_D' + str(
                                args.D) + 'All' + '_' +timestamp + '.txt')
    test_filename_all = os.path.join(DATA_PATH,  'attention_data_test_' + str(args.channel) + '_lr_' + str(args.enc_lr) + '_D' + str(args.D) + 'All' + '_' +timestamp + '.txt')
    data_file_all = open(filename_all, 'a')
    # data_test_file_all = open(test_filename_all, 'a')
    for blocklen in (args.block_len):
        for coderate_k, coderate_n in zip( args.code_rate_k, args.code_rate_n):
            for mod_type in (args.modtype):
                print('\n \n ###############################################################################################')
                print('Training Model for :',
                      'block length =>',blocklen,
                      'coderate_k =>', coderate_k,
                      'coderate_n =>', coderate_n,
                      'modulation_type =>', mod_type)
                prefix = 'bl_' + str(blocklen) + '_' + '_k_' + str(coderate_k) +  '_n_' + str(coderate_n)+  '_mod_' + str(mod_type)
                for path in (os.path.join(DATA_PATH,prefix), os.path.join(MODEL_PATH,prefix)):
                    if not os.path.isdir(path):
                        os.makedirs(path)
                filename = os.path.join(DATA_PATH,prefix,
                                        'attention_data_' + str(args.channel) + '_lr_' + str(args.enc_lr) + '_D' + str(
                                            args.D) +prefix + '_' + str(args.num_block) + timestamp + '.txt')
                test_filename = os.path.join(DATA_PATH, prefix,'attention_data_test_' + str(args.channel) + '_lr_' + str(
                    args.enc_lr) + '_D' + str(
                    args.D)+prefix + '_' + str(args.num_block) + '_' + timestamp + '.txt')
                attnFilename.append(filename)

                store_files = ( filename, test_filename)

                train_model(args,timestamp,store_files,start_epoch,use_cuda,blocklen,coderate_k,coderate_n,mod_type)
                # f_data = open(filename, 'r')
                # f_data.seek(0)
                # data_file_all.write(f_data.read())
                # f_test = open(test_filename, 'r')
                # f_test.seek(0)
                # data_test_file_all.write(f_test.read())
                data_file_all.write(
                    str(blocklen) +' ' + str(coderate_k) + ' '+ str(coderate_n) + ' '+ str(float(coderate_k/coderate_n)) +' '+ mod_type +' '+str(filename) + ' ' + str(test_filename) + "\n")
    # data_test_file_all.close()
    data_file_all.close()
    get_plots(args.PLOT_PATH,filename_all)
    # lr = 0.01
    # filenamed1 = './data/data_awgn_lr_' + str(lr) + '_D1_10000.txt'
    # filenamed10 = './data/data_awgn_lr_' + str(lr) + '_D1_10000.txt'
    # plot_attn(lr, 1, filenamed1, filenamed10)
    # plot(filenamed1, legend)
    # plot_snr(filenamed1, legend_snr)
    # plot_attn(lr, 10, filenamed1, filenamed10)
    # plot(filenamed10, legend)
    # plot_snr(filenamed10, legend_snr)
    # print("Training Time: {}s".format(time.time() - start_time))




