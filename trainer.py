import os

import torch
import time
import torch.nn.functional as F
from torch import device, optim
from numpy import arange

# choose support channels
from channel_ae import Channel_AE
from decoders import DEC_LargeCNN as DEC
from encoders import ENC_interCNN as ENC

from utils import snr_sigma2db, snr_db2sigma, code_power, errors_ber_pos, errors_ber, errors_bler, generate_noise, \
    customized_loss, generate_noise_SNR, get_modem, generate_noise_SNR_Sim

import numpy as np
from numpy.random import mtrand

######################################################################################
#
# Trainer, validation, and test for AE code design
#
######################################################################################

def train_model(args,timestamp,store_files,start_epoch=1,use_cuda=False,blocklen =10,coderate_k=1,coderate_n=3,channel='awgn'):

    datafile = store_files[0]
    testfile = store_files[1]
    #################################################
    # Setup Channel AE: Encoder, Decoder, Channel
    #################################################

    encoder = ENC(args,coderate_k,coderate_n)
    decoder = DEC(args,blocklen,coderate_k,coderate_n)
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Channel_AE(args,encoder,decoder).to(device)
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

    snrs = np.arange(0, 100, 5)
    for epoch in range(start_epoch, args.num_epoch + 1):

            epoch_start_time = time.time()
            # save model per epoch
            prefix = 'bl_' + str(blocklen) + '_' + '_k_' + str(coderate_k) +  '_n_' + str(coderate_n)+  '_chl_' + str(channel)
            modelpath = os.path.join(args.MODEL_PATH, prefix,
                                     'attention_model_' + str(epoch) + '_' + str(channel) + '_lr_' + str(
                                         args.enc_lr) + '_D' + str(args.D) + prefix + '_' + str(
                                         args.num_block) + '_' + timestamp + '.pt')
            if args.joint_train == 1 and args.encoder not in ['Turbo_rate3_lte', 'Turbo_rate3_757']:
                for idx in range(args.num_train_enc + args.num_train_dec):
                    train(epoch, model, general_optimizer, args,channel=channel,blocklen =blocklen,coderate_k=coderate_k,coderate_n=coderate_n, use_cuda=use_cuda, mode='encoder')

            else:
                if args.num_train_enc > 0 and args.encoder not in ['Turbo_rate3_lte', 'Turbo_rate3_757']:
                    for idx in range(args.num_train_enc):
                        train(epoch, model, enc_optimizer, args, channel=channel,blocklen =blocklen,coderate_k=coderate_k,coderate_n=coderate_n,use_cuda=use_cuda, mode='encoder')

                if args.num_train_dec > 0:
                    for idx in range(args.num_train_dec):
                        train(epoch, model, dec_optimizer, args,channel=channel,blocklen =blocklen,coderate_k=coderate_k,coderate_n=coderate_n, use_cuda=use_cuda, mode='decoder')

            this_loss, this_ber = validate(model, general_optimizer, args,channel=channel,blocklen =blocklen,coderate_k=coderate_k,coderate_n=coderate_n, use_cuda=use_cuda)
            report_loss.append(this_loss)
            report_ber.append(this_ber)


    if args.print_test_traj == True:
        print('test loss trajectory for channel '+channel+':', report_loss)
        print('test ber trajectory '+channel+':', report_ber)
        print('total epoch '+channel+':', args.num_epoch)

        #################################################
        # Testing Processes
        #################################################
    torch.save(model.state_dict(),modelpath)
    print('saved model', modelpath)

    if args.is_variable_block_len:
        print('testing block length '+channel+':', args.block_len_low)
        test(model, args,testfile,channel=channel,blocklen =blocklen,coderate_k=coderate_k,coderate_n=coderate_n, block_len=args.block_len_low, use_cuda=use_cuda)
        print('testing block length '+channel+':', args.block_len)
        test(model, args,testfile,channel=channel,blocklen =blocklen,coderate_k=coderate_k,coderate_n=coderate_n, block_len=args.block_len, use_cuda=use_cuda)
        print('testing block length '+channel+':', args.block_len_high)
        test(model, args,testfile,channel=channel,blocklen =blocklen,coderate_k=coderate_k,coderate_n=coderate_n, block_len=args.block_len_high, use_cuda=use_cuda)

    else:
        test(model, args,testfile,channel=channel,blocklen =blocklen,coderate_k=coderate_k,coderate_n=coderate_n, use_cuda=use_cuda)

def train_model_mod(args,timestamp,store_files,start_epoch=1,use_cuda=False,blocklen =10,coderate_k=1,coderate_n=3,channel='awgn',mod_rate=2):

    datafile = store_files[0]
    testfile = store_files[1]
    #################################################
    # Setup Channel AE: Encoder, Decoder, Channel
    #################################################


    device = torch.device("cuda" if use_cuda else "cpu")

    # setup interleaver.
    if args.is_interleave == 1:           # fixed interleaver.
        seed = np.random.randint(0, 1)
        rand_gen = mtrand.RandomState(seed)
        p_array1 = rand_gen.permutation(arange(blocklen))
        p_array2 = rand_gen.permutation(arange(blocklen))

    elif args.is_interleave == 0:
        p_array1 = range(blocklen)   # no interleaver.
        p_array2 = range(blocklen)   # no interleaver.
    else:
        seed = np.random.randint(0, args.is_interleave)
        rand_gen = mtrand.RandomState(seed)
        p_array1 = rand_gen.permutation(arange(blocklen))
        seed = np.random.randint(0, args.is_interleave)
        rand_gen = mtrand.RandomState(seed)
        p_array2 = rand_gen.permutation(arange(blocklen))

    print('using random interleaver', p_array1, p_array2)

    if args.encoder == 'turboae_2int' and args.decoder == 'turboae_2int':
        encoder = ENC(args, p_array1, p_array2,coderate_k,coderate_n,blocklen)
        decoder = DEC(args, p_array1, p_array2,coderate_k,coderate_n,blocklen)
    else:
        encoder = ENC(args, p_array1,coderate_k,coderate_n,blocklen)
        decoder = DEC(args, p_array1,coderate_k,coderate_n,blocklen)

    # modulation and demodulations.
    from modulations import Modulation, DeModulation

    modulator = Modulation(args,blocklen,coderate_k,coderate_n,mod_rate)
    demodulator = DeModulation(args,blocklen,coderate_k,coderate_n,mod_rate)

    # choose support channels
    from channel_ae import Channel_ModAE
    model = Channel_ModAE(args, encoder, decoder, modulator, demodulator).to(device)

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
    # Setup Optimizers, only Adam and Lookahead for now.
    ##################################################################

    if args.optimizer == 'lookahead':
        print('Using Lookahead Optimizers')
        from optimizers import Lookahead
        lookahead_k = 5
        lookahead_alpha = 0.5
        if args.num_train_enc != 0 and args.encoder not in ['Turbo_rate3_lte',
                                                            'Turbo_rate3_757']:  # no optimizer for encoder
            enc_base_opt = optim.Adam(model.enc.parameters(), lr=args.enc_lr)
            enc_optimizer = Lookahead(enc_base_opt, k=lookahead_k, alpha=lookahead_alpha)

        if args.num_train_dec != 0:
            dec_base_opt = optim.Adam(filter(lambda p: p.requires_grad, model.dec.parameters()), lr=args.dec_lr)
            dec_optimizer = Lookahead(dec_base_opt, k=lookahead_k, alpha=lookahead_alpha)

        general_base_opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.dec_lr)
        general_optimizer = Lookahead(general_base_opt, k=lookahead_k, alpha=lookahead_alpha)

    else:  # Adam, SGD, etc....
        if args.optimizer == 'adam':
            OPT = optim.Adam
        elif args.optimizer == 'sgd':
            OPT = optim.SGD
        else:
            OPT = optim.Adam

        if args.num_train_enc != 0 and args.encoder not in ['Turbo_rate3_lte',
                                                            'Turbo_rate3_757']:  # no optimizer for encoder
            enc_optimizer = OPT(model.enc.parameters(), lr=args.enc_lr)

        if args.num_train_dec != 0:
            dec_optimizer = OPT(filter(lambda p: p.requires_grad, model.dec.parameters()), lr=args.dec_lr)

        if args.num_train_mod != 0:
            mod_optimizer = OPT(filter(lambda p: p.requires_grad, model.mod.parameters()), lr=args.mod_lr)

        if args.num_train_demod != 0:
            demod_optimizer = OPT(filter(lambda p: p.requires_grad, model.demod.parameters()), lr=args.demod_lr)

        general_optimizer = OPT(filter(lambda p: p.requires_grad, model.parameters()), lr=args.dec_lr)

    #################################################
    # Training Processes
    #################################################
    report_loss, report_ber, report_bler = [], [], []

    snrs = np.arange(0, 100, 5)
    for epoch in range(start_epoch, args.num_epoch + 1):

            epoch_start_time = time.time()
            # save model per epoch
            prefix = 'bl_' + str(blocklen) + '_' + '_k_' + str(coderate_k) +  '_n_' + str(coderate_n)+  '_chl_' + str(channel)+'_mod_QAM' + str(mod_rate)
            modelpath = os.path.join(args.MODEL_PATH, prefix,
                                     'attention_model_' + str(epoch) + '_' + str(channel) + '_lr_' + str(
                                         args.enc_lr) + '_D' + str(args.D) + prefix + '_' + str(
                                         args.num_block) + '_' + timestamp + '.pt')
            if args.joint_train == 1 and args.encoder not in ['Turbo_rate3_lte', 'Turbo_rate3_757']:
                for idx in range(args.num_train_enc + args.num_train_dec):
                    train(epoch, model, general_optimizer, args,channel=channel,blocklen =blocklen,coderate_k=coderate_k,coderate_n=coderate_n, use_cuda=use_cuda, mode='encoder',mod_rate=mod_rate)

            else:
                if args.num_train_enc > 0 and args.encoder not in ['Turbo_rate3_lte', 'Turbo_rate3_757']:
                    for idx in range(args.num_train_enc):
                        train(epoch, model, enc_optimizer, args, channel=channel,blocklen =blocklen,coderate_k=coderate_k,coderate_n=coderate_n,use_cuda=use_cuda, mode='encoder',mod_rate=mod_rate)

                if args.num_train_dec > 0:
                    for idx in range(args.num_train_dec):
                        train(epoch, model, dec_optimizer, args,channel=channel,blocklen =blocklen,coderate_k=coderate_k,coderate_n=coderate_n, use_cuda=use_cuda, mode='decoder',mod_rate=mod_rate)

                if args.num_train_mod > 0 :
                    for idx in range(args.num_train_mod):
                        train(epoch, model, mod_optimizer, args, channel=channel,blocklen =blocklen,coderate_k=coderate_k,coderate_n=coderate_n,use_cuda=use_cuda, mode='encoder',mod_rate=mod_rate)

                if args.num_train_demod > 0:
                    for idx in range(args.num_train_demod):
                        train(epoch, model, demod_optimizer, args,channel=channel,blocklen =blocklen,coderate_k=coderate_k,coderate_n=coderate_n, use_cuda=use_cuda, mode='decoder',mod_rate=mod_rate)

            this_loss, this_ber = validate(model, general_optimizer, args,channel=channel,blocklen =blocklen,coderate_k=coderate_k,coderate_n=coderate_n, use_cuda=use_cuda,mod_rate=mod_rate)
            report_loss.append(this_loss)
            report_ber.append(this_ber)


    if args.print_test_traj == True:
        print('test loss trajectory for channel '+channel+':', report_loss)
        print('test ber trajectory '+channel+':', report_ber)
        print('total epoch '+channel+':', args.num_epoch)

        #################################################
        # Testing Processes
        #################################################
    torch.save(model.state_dict(),modelpath)
    print('saved model', modelpath)

    if args.is_variable_block_len:
        print('testing block length '+channel+':', args.block_len_low)
        test(model, args,testfile,channel=channel,blocklen =blocklen,coderate_k=coderate_k,coderate_n=coderate_n, block_len=args.block_len_low, use_cuda=use_cuda,mod_rate=mod_rate)
        print('testing block length '+channel+':', args.block_len)
        test(model, args,testfile,channel=channel,blocklen =blocklen,coderate_k=coderate_k,coderate_n=coderate_n, block_len=args.block_len, use_cuda=use_cuda,mod_rate=mod_rate)
        print('testing block length '+channel+':', args.block_len_high)
        test(model, args,testfile,channel=channel,blocklen =blocklen,coderate_k=coderate_k,coderate_n=coderate_n, block_len=args.block_len_high, use_cuda=use_cuda,mod_rate=mod_rate)

    else:
        test(model, args,testfile,channel=channel,blocklen =blocklen,coderate_k=coderate_k,coderate_n=coderate_n, use_cuda=use_cuda,mod_rate=mod_rate)

def train(epoch, model, optimizer, args,channel='awgn',blocklen =10,coderate_k=1,coderate_n=3 ,use_cuda = False, verbose = True, mode = 'encoder',mod_rate=2):

    device = torch.device("cuda" if use_cuda else "cpu")

    model.train()
    start_time = time.time()
    train_loss = 0.0
    k_same_code_counter = 0


    for batch_idx in range(int(args.num_block/args.batch_size)):


        # if args.is_variable_block_len:
        #     block_len = np.random.randint(args.block_len_low, args.block_len_high)
        # else:
        #     block_len = args.block_len

        optimizer.zero_grad()

        if args.is_k_same_code and mode == 'encoder':
            if batch_idx == 0:
                k_same_code_counter += 1
                X_train    = torch.randint(0, 2, (args.batch_size, blocklen,coderate_k), dtype=torch.float)
            elif k_same_code_counter == args.k_same_code:
                k_same_code_counter = 1
                X_train    = torch.randint(0, 2, (args.batch_size, blocklen, coderate_k), dtype=torch.float)
            else:
                k_same_code_counter += 1
        else:
            X_train    = torch.randint(0, 2, (args.batch_size, blocklen, coderate_k), dtype=torch.float)

        noise_shape = (args.batch_size,int( blocklen* coderate_n/mod_rate),mod_rate)
        # train encoder/decoder with different SNR... seems to be a good practice.
        if mode == 'encoder':
            fwd_noise  = generate_noise(noise_shape, args, snr_low=args.train_enc_channel_low, snr_high=args.train_enc_channel_high, mode = 'encoder')
        else:
            fwd_noise  = generate_noise(noise_shape, args, snr_low=args.train_dec_channel_low, snr_high=args.train_dec_channel_high, mode = 'decoder')

        X_train, fwd_noise = X_train.to(device), fwd_noise.to(device)

        output, code = model(X_train, fwd_noise,channel,blocklen)
        output = torch.clamp(output, 0.0, 1.0)

        if mode == 'encoder':
            loss = customized_loss(output, X_train, args, noise=fwd_noise, code = code)

        else:
            loss = customized_loss(output, X_train, args, noise=fwd_noise, code = code)
            #loss = F.binary_cross_entropy(output, X_train)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    end_time = time.time()
    train_loss = train_loss /(args.num_block/args.batch_size)
    if verbose:
        print('====> Epoch: {} Average loss: {:.8f}'.format(epoch, train_loss), \
            ' running time', str(end_time - start_time))

    return train_loss

def validate(model, optimizer, args,channel='awgn',blocklen =10,coderate_k=1,coderate_n=3 , use_cuda = False, verbose = True,mod_rate=2):

    device = torch.device("cuda" if use_cuda else "cpu")

    model.eval()
    test_bce_loss, test_custom_loss, test_ber= 0.0, 0.0, 0.0

    with torch.no_grad():
        num_test_batch = int(args.num_block/args.batch_size * args.test_ratio)
        for batch_idx in range(num_test_batch):
            X_test     = torch.randint(0, 2, (args.batch_size, blocklen, coderate_k), dtype=torch.float)
            noise_shape = (args.batch_size,int( blocklen* coderate_n/mod_rate), mod_rate)
            fwd_noise  = generate_noise(noise_shape, args,
                                        snr_low=args.train_enc_channel_low,
                                        snr_high=args.train_enc_channel_low)

            X_test, fwd_noise= X_test.to(device), fwd_noise.to(device)

            optimizer.zero_grad()
            output, codes = model(X_test, fwd_noise,channel,blocklen)

            output = torch.clamp(output, 0.0, 1.0)

            output = output.detach()
            X_test = X_test.detach()

            test_bce_loss += F.binary_cross_entropy(output, X_test)
            test_custom_loss += customized_loss(output, X_test, noise = fwd_noise, args = args, code = codes)
            test_ber  += errors_ber(output,X_test)


    test_bce_loss /= num_test_batch
    test_custom_loss /= num_test_batch
    test_ber  /= num_test_batch

    if verbose:
        print('====> Test set BCE loss', float(test_bce_loss),
              'Custom Loss',float(test_custom_loss),
              'with ber ', float(test_ber),
        )

    report_loss = float(test_bce_loss)
    report_ber  = float(test_ber)

    return report_loss, report_ber

def test(model, args,filename ,channel='awgn',blocklen =10,coderate_k=1,coderate_n=3 , block_len = 'default',use_cuda = False,mod_rate=2):

    device = torch.device("cuda" if use_cuda else "cpu")
    model.eval()
    data_file = open(filename, 'a')
    # if block_len == 'default':
    #     block_len = args.block_len
    # else:
    #     pass

    # Precomputes Norm Statistics.
    if args.precompute_norm_stats:
        with torch.no_grad():
            num_test_batch = int(args.num_block/(args.batch_size)* args.test_ratio)
            for batch_idx in range(num_test_batch):
                X_test = torch.randint(0, 2, (args.batch_size, blocklen, coderate_k), dtype=torch.float)
                X_test = X_test.to(device)
                _      = model.enc(X_test)
            print('Pre-computed norm statistics mean ',model.enc.mean_scalar, 'std ', model.enc.std_scalar)

    ber_res, bler_res = [], []
    ber_res_punc, bler_res_punc = [], []
    snr_interval = (args.snr_test_end - args.snr_test_start)* 1.0 /  (args.snr_points)
    snrs = [snr_interval* item + args.snr_test_start for item in range(args.snr_points)]
    print('SNRS', snrs)
    sigmas = snrs

    for sigma, this_snr in zip(sigmas, snrs):
        test_ber, test_bler = .0, .0
        with torch.no_grad():
            num_test_batch = int(args.num_block/(args.batch_size))
            for batch_idx in range(num_test_batch):
                X_test     = torch.randint(0, 2, (args.batch_size, blocklen, coderate_k), dtype=torch.float)
                noise_shape = (args.batch_size,int( blocklen* coderate_n/mod_rate), mod_rate)
                fwd_noise  = generate_noise(noise_shape, args, test_sigma=sigma,snr_low=args.train_enc_channel_low, snr_high=args.train_enc_channel_high,)

                X_test, fwd_noise= X_test.to(device), fwd_noise.to(device)

                X_hat_test, the_codes = model(X_test, fwd_noise,channel,blocklen)


                test_ber  += errors_ber(X_hat_test,X_test)
                test_bler += errors_bler(X_hat_test,X_test)

                if batch_idx == 0:
                    test_pos_ber = errors_ber_pos(X_hat_test,X_test)
                    codes_power  = code_power(the_codes)
                else:
                    test_pos_ber += errors_ber_pos(X_hat_test,X_test)
                    codes_power  += code_power(the_codes)
            if args.print_pos_power:
                print('code power', codes_power/num_test_batch)
            if args.print_pos_ber:
                res_pos = test_pos_ber/num_test_batch
                res_pos_arg = np.array(res_pos.cpu()).argsort()[::-1]
                res_pos_arg = res_pos_arg.tolist()
                print('positional ber', res_pos)
                print('positional argmax',res_pos_arg)
            try:
                test_ber_punc, test_bler_punc = .0, .0
                for batch_idx in range(num_test_batch):
                    X_test     = torch.randint(0, 2, (args.batch_size, blocklen, coderate_k), dtype=torch.float)
                    fwd_noise  = generate_noise(X_test.shape, args, test_sigma=sigma,snr_low=args.train_enc_channel_low, snr_high=args.train_enc_channel_high,)
                    X_test, fwd_noise= X_test.to(device), fwd_noise.to(device)

                    X_hat_test, the_codes = model(X_test, fwd_noise,channel,blocklen)

                    test_ber_punc  += errors_ber(X_hat_test,X_test, positions = res_pos_arg[:args.num_ber_puncture])
                    test_bler_punc += errors_bler(X_hat_test,X_test, positions = res_pos_arg[:args.num_ber_puncture])

                    if batch_idx == 0:
                        test_pos_ber = errors_ber_pos(X_hat_test,X_test)
                        codes_power  = code_power(the_codes)
                    else:
                        test_pos_ber += errors_ber_pos(X_hat_test,X_test)
                        codes_power  += code_power(the_codes)
            except:
                print('no pos BER specified.')

        test_ber  /= num_test_batch
        test_bler /= num_test_batch
        print('Test SNR',this_snr ,'with ber ', float(test_ber), 'with bler', float(test_bler))
        ber_res.append(float(test_ber))
        bler_res.append( float(test_bler))

        try:
            test_ber_punc  /= num_test_batch
            test_bler_punc /= num_test_batch
            print('Punctured Test SNR',this_snr ,'with ber ', float(test_ber_punc), 'with bler', float(test_bler_punc))
            ber_res_punc.append(float(test_ber_punc))
            bler_res_punc.append( float(test_bler_punc))
        except:
            print('No puncturation is there.')

        data_file.write(str(blocklen) + ' ' + str(coderate_k) + ' ' + str(coderate_n)
                    + ' ' + str(this_snr) + ' ' + str(float(test_ber))+' '+str(float(test_ber_punc))+ "\n")
    data_file.close()
    print('final results on SNRs ', snrs)
    print('BER', ber_res)
    print('BLER', bler_res)
    print('final results on punctured SNRs ', snrs)
    print('BER', ber_res_punc)
    print('BLER', bler_res_punc)

    # compute adjusted SNR. (some quantization might make power!=1.0)
    enc_power = 0.0
    with torch.no_grad():
        for idx in range(num_test_batch):
            X_test     = torch.randint(0, 2, (args.batch_size, blocklen, coderate_k), dtype=torch.float)
            X_test     = X_test.to(device)
            X_code     = model.enc(X_test)
            enc_power +=  torch.std(X_code)
    enc_power /= float(num_test_batch)
    print('encoder power is',enc_power)
    adj_snrs = [snr_sigma2db(snr_db2sigma(item)/enc_power) for item in snrs]
    print('adjusted SNR should be',adj_snrs)

