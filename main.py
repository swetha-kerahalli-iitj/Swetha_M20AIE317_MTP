import torch
import torch.optim as optim
import sys, os
import time
from get_args import get_args
from plot import plot_attn, plot, plot_snr,legend, legend_snr
from trainer import train, validate, test

from encoder import ENC
from decoder import DEC

BASE_PATH = r'C:\WorkSpace\Swetha_M20AIE317_MTP'
LOG_PATH = os.path.join(BASE_PATH, r'logs_test')
DATA_PATH = os.path.join(BASE_PATH, r'data_test')
MODEL_PATH = os.path.join(BASE_PATH, r'model_test')
for path in (LOG_PATH,DATA_PATH,MODEL_PATH):
    if not os.path.isdir(path):
        os.makedirs(path)
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

    # put all printed things to log file
    if args.init_nw_weight == 'default':
        start_epoch = 1
        timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    else:
        start_epoch = int(args.init_nw_weight.split('_')[2]) + 1
        timestamp = args.init_nw_weight.split('_')[8].split('.')[0]
    formatfile = 'attention_log_' + str(args.channel) + '_lr_' + str(args.enc_lr) + '_D' + str(args.D) + '_' + str(
        args.num_block) + '_' + timestamp + '.txt'
    logfilename = os.path.join(LOG_PATH, formatfile)
    logfile = open(logfilename, 'a')
    sys.stdout = Logger(logfilename, sys.stdout)

    print(args)

    filename = os.path.join(DATA_PATH, 'attention_data_' + str(args.channel) + '_lr_' + str(args.enc_lr) + '_D' + str(
        args.D) + '_' + str(args.num_block) + '_' + timestamp + '.txt')

    attnFilename.append(filename)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("use_cuda: ", use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    #################################################
    # Setup Channel AE: Encoder, Decoder, Channel
    #################################################

    encoder = ENC(args)
    decoder = DEC(args)

    # choose support channels
    from channel_ae import Channel_AE

    model = Channel_AE(args, encoder, decoder).to(device)

    # weight loading
    if args.init_nw_weight == 'default':
        pass

    else:
        pretrained_model = torch.load(args.init_nw_weight)

        try:
            model.load_state_dict(pretrained_model.state_dict(), strict=False)

        except:
            model.load_state_dict(pretrained_model, strict=False)

        model.args = args

    print(model)

    ##################################################################
    # Setup Optimizers
    ##################################################################

    OPT = optim.Adam

    if args.num_train_enc != 0:  # no optimizer for encoder
        enc_optimizer = OPT(model.enc.parameters(), lr=args.enc_lr)

    if args.num_train_dec != 0:
        dec_optimizer = OPT(filter(lambda p: p.requires_grad, model.dec.parameters()), lr=args.dec_lr)

    general_optimizer = OPT(filter(lambda p: p.requires_grad, model.parameters()), lr=args.dec_lr)

    #################################################
    # Training Processes
    #################################################
    report_loss, report_ber, report_bler = [], [], []

    for epoch in range(start_epoch, args.num_epoch + 1):
        epoch_start_time = time.time()
        if args.num_train_enc > 0:
            for idx in range(args.num_train_enc):
                train(epoch, model, enc_optimizer, args, use_cuda=use_cuda, mode='encoder')

        if args.num_train_dec > 0:
            for idx in range(args.num_train_dec):
                train(epoch, model, dec_optimizer, args, use_cuda=use_cuda, mode='decoder')

        this_loss, this_ber, this_bler = validate(model, general_optimizer, args, use_cuda=use_cuda)

        report_loss.append(this_loss)
        report_ber.append(this_ber)
        report_bler.append(this_bler)

        data_file = open(filename, 'a')
        data_file.write(str(epoch) + ' ' + str(this_loss) + ' ' + str(this_ber) + ' ' + str(this_bler) + "\n")
        data_file.close()

        # save model per epoch
        modelpath = os.path.join(MODEL_PATH, 'attention_model_' + str(epoch) + '_' + str(args.channel) + '_lr_' + str(
            args.enc_lr) + '_D' + str(args.D) + '_' + str(args.num_block) + '_' + timestamp + '.pt')
        torch.save(model.state_dict(), modelpath)
        print('saved model', modelpath)
        print("each epoch training time: {}s".format(time.time() - epoch_start_time))

    if args.print_test_traj == True:
        print('test loss trajectory', report_loss)
        print('test ber trajectory', report_ber)
        print('test bler trajectory', report_bler)
        print('total epoch', args.num_epoch)

    #################################################
    # Testing Processes
    #################################################

    modelpath = os.path.join(MODEL_PATH,
                             'attention_model_' + str(args.channel) + '_lr_' + str(args.enc_lr) + '_D' + str(
                                 args.D) + '_' + str(args.num_block) + '.pt')

    torch.save(model.state_dict(), modelpath)
    print('saved model', modelpath)

    if args.is_variable_block_len:
        print('testing block length', args.block_len_low)
        test(model, args, block_len=args.block_len_low, use_cuda=use_cuda)
        print('testing block length', args.block_len)
        test(model, args, block_len=args.block_len, use_cuda=use_cuda)
        print('testing block length', args.block_len_high)
        test(model, args, block_len=args.block_len_high, use_cuda=use_cuda)

    else:
        test(model, args, use_cuda=use_cuda)

    # get_plots(attnFilename,args.enc_lr)
    lr = 0.01
    filenamed1 = './data/data_awgn_lr_' + str(lr) + '_D1_10000.txt'
    filenamed10 = './data/data_awgn_lr_' + str(lr) + '_D1_10000.txt'
    plot_attn(lr, 1, filenamed1, filenamed10)
    plot(filenamed1, legend)
    plot_snr(filenamed1, legend_snr)
    plot_attn(lr, 10, filenamed1, filenamed10)
    plot(filenamed10, legend)
    plot_snr(filenamed10, legend_snr)
    print("Training Time: {}s".format(time.time() - start_time))




