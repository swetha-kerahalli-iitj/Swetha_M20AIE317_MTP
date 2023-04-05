import os

import torch
import time
import torch.nn.functional as F
from torch import device, optim

# choose support channels
from channel_ae import Channel_AE
from decoder import DEC
from encoder import ENC

from utils import snr_sigma2db, snr_db2sigma, code_power, errors_ber_pos, errors_ber, errors_bler, generate_noise, \
    customized_loss, generate_noise_SNR, get_modem, generate_noise_SNR_Sim

import numpy as np
from numpy.random import mtrand

######################################################################################
#
# Trainer, validation, and test for AE code design
#
######################################################################################
def train_model(args,timestamp,store_files,start_epoch=1,use_cuda=False,blocklen =10,coderate_k=1,coderate_n=3,mod_type='QAM16'):
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

    for epoch in range(start_epoch, args.num_epoch + 1):
        epoch_start_time = time.time()
        # save model per epoch
        prefix = 'bl_' + str(blocklen) + '_' + '_k_' + str(coderate_k) +  '_n_' + str(coderate_n)+  '_mod_' + str(mod_type)
        modelpath = os.path.join(args.MODEL_PATH, prefix,
                                 'attention_model_' + str(epoch) + '_' + str(args.channel) + '_lr_' + str(
                                     args.enc_lr) + '_D' + str(args.D) + prefix + '_' + str(
                                     args.num_block) + '_' + timestamp + '.pt')
        if args.num_train_enc > 0:
            for idx in range(args.num_train_enc):
                train(epoch, model, enc_optimizer, args,blocklen,coderate_k,coderate_n,mod_type, use_cuda=use_cuda, mode='encoder')

        if args.num_train_dec > 0:
            for idx in range(args.num_train_dec):
                train(epoch, model, dec_optimizer, args,blocklen,coderate_k,coderate_n,mod_type,use_cuda=use_cuda, mode='decoder')

        this_loss, this_ber, this_bler = validate(model, general_optimizer, args,blocklen,coderate_k,coderate_n,mod_type ,use_cuda=use_cuda)

        report_loss.append(this_loss)
        report_ber.append(this_ber)
        report_bler.append(this_bler)

        data_file = open(datafile, 'a')
        data_file.write(str(epoch) + ' ' + str(this_loss) + ' ' + str(this_ber) + ' ' + str(this_bler) + "\n")
        data_file.close()


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

    torch.save(model.state_dict(), modelpath)
    print('saved model', modelpath)

    if args.is_variable_block_len:
        print('testing block length', args.block_len_low)
        test(model, testfile, args,args.block_len_low,coderate_k,coderate_n, use_cuda=use_cuda)
        print('testing block length', blocklen)
        test(model, testfile, args,blocklen,coderate_k,coderate_n, use_cuda=use_cuda)
        print('testing block length', args.block_len_high)
        test(model, testfile, args, args.block_len_high,coderate_k,coderate_n, use_cuda=use_cuda)

    else:
        test(model, testfile, args,blocklen,coderate_k,coderate_n,mod_type, use_cuda=use_cuda)
def train_noise (model,optimizer,X_train,args,SNR,noise_shape,use_cuda,noise_type,mode,coderate_k,coderate_n,mod_type='QAM16'):
    device = torch.device("cuda" if use_cuda else "cpu")
    if mode == 'encoder':

        fwd_noise, received_data, encoded_input, input_msg, mod, parity_h, parity_g = generate_noise_SNR(SNR, noise_shape,args, noise_type,coderate_k,coderate_n,mod_type)

    else:
        fwd_noise, received_data, encoded_input, input_msg, mod, parity_h, parity_g = generate_noise_SNR( SNR, noise_shape,args,noise_type,coderate_k,coderate_n,mod_type)

    noise = fwd_noise.to(device)
    fwd_noise = noise.real.float()

    # print('train:',
    #       'fwd_noise.shape =>',fwd_noise.shape,
    #       'noise_shape =>', noise_shape,
    #       'X_train shape =>', X_train.shape,
    #       'coderate_k  =>',coderate_k ,
    #       'coderate_n  =>',coderate_n
    #       )
    output, code = model(X_train, fwd_noise)
    output = torch.clamp(output, 0.0, 1.0)


    if mode == 'encoder':
        loss = customized_loss(output, X_train, args, noise=fwd_noise, code = mode)
    else:
        loss = customized_loss(output, X_train, args, noise=fwd_noise, code = mode)

    loss.backward()
    train_loss = loss.item()
    optimizer.step()
    return train_loss

def train(epoch, model, optimizer, args,block_len=10,coderate_k=1,coderate_n=3,mod_type='QAM16', use_cuda = False, verbose = True, mode = 'encoder'):

    device = torch.device("cuda" if use_cuda else "cpu")

    model.train()
    start_time = time.time()
    train_loss_awgn,train_loss_ray,train_loss_rici = 0.0, 0.0, 0.0

    for batch_idx in range(int(args.num_block/args.batch_size)):

        optimizer.zero_grad()
        if mod_type != "LDPC" or mod_type != "POLAR":
            mod = get_modem( mod_type)
            X_train    = torch.randint(0, mod.M, (args.batch_size, block_len, coderate_k), dtype=torch.float)
            X_train = X_train.to(device)
        else:
            X_train = torch.randint(0, 2, (args.batch_size, block_len, coderate_k), dtype=torch.float)
            X_train = X_train.to(device)
        noise_shape = (args.batch_size, block_len, coderate_n)
        # train encoder/decoder with different SNR... seems to be a good practice.
        SNR = args.train_enc_channel_low
        code_rate = coderate_k / coderate_n
        train_loss_awgn += train_noise(model,optimizer,X_train,args,SNR,noise_shape,use_cuda,"AWGN",mode,coderate_k,coderate_n,mod_type)
        train_loss_ray += train_noise(model,optimizer,X_train,args,SNR,noise_shape,use_cuda,"Rayleigh",mode,coderate_k,coderate_n,mod_type)
        train_loss_rici += train_noise(model,optimizer,X_train,args,SNR,noise_shape,use_cuda,"Rician",mode,coderate_k,coderate_n,mod_type)

    end_time = time.time()
    train_loss_awgn = train_loss_awgn /(args.num_block/args.batch_size)
    train_loss_ray = train_loss_ray / (args.num_block / args.batch_size)
    train_loss_rici = train_loss_rici / (args.num_block / args.batch_size)
    if verbose:
        print('====> Epoch: {} Average loss AWGN: {:.8f} loss Rayleigh: {:.8f} loss Rician: {:.8f} '.format(epoch, train_loss_awgn,train_loss_ray,train_loss_rici), \
            ' running time', str(end_time - start_time))

    return train_loss_awgn,train_loss_ray,train_loss_rici

def validate_noise(model, optimizer,X_test, args,SNR,noise_shape,use_cuda,noise_type,coderate_k=1,coderate_n=3,mod_type="QAM16"):
    device = torch.device("cuda" if use_cuda else "cpu")
    test_bce_loss, test_custom_loss, test_ber, test_bler = 0.0, 0.0, 0.0, 0.0
    fwd_noise, received_data, encoded_input, input_msg, mod, parity_h, parity_g = generate_noise_SNR(SNR, noise_shape, args,noise_type ,coderate_k,coderate_n,mod_type)
    # print('train:',
    #             'noise_shape =>',noise_shape,
    #             'coderate_k  =>',coderate_k ,
    #             'coderate_n  =>',coderate_n,
    #             'fwd_noise =>',fwd_noise.shape
    #             )
    noise = fwd_noise.to(device)
    X_test, fwd_noise = X_test.to(device), fwd_noise.to(device)
    optimizer.zero_grad()

    output, codes = model(X_test, fwd_noise)

    output = torch.clamp(output, 0.0, 1.0)

    output = output.detach()
    X_test = X_test.detach()

    test_bce_loss = F.binary_cross_entropy(output, X_test)
    test_custom_loss = customized_loss(output, X_test, noise=fwd_noise, args=args, code=codes)
    test_ber = errors_ber(output, X_test)
    test_bler = errors_bler(output, X_test)
    return test_bce_loss,test_custom_loss,test_ber,test_bler
def validate(model, optimizer, args, block_len=10, coderate_k=1, coderate_n=3,mod_type="QAM16", use_cuda = False, verbose = True,
            ):

    model.eval()
    test_bce_loss_awgn, test_custom_loss_awgn, test_ber_awgn, test_bler_awgn = 0.0, 0.0, 0.0, 0.0
    test_bce_loss_ray, test_custom_loss_ray, test_ber_ray, test_bler_ray = 0.0, 0.0, 0.0, 0.0
    test_bce_loss_rici, test_custom_loss_rici, test_ber_rici, test_bler_rici = 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        num_test_batch = int(args.num_block/args.batch_size * args.test_ratio)
        for batch_idx in range(num_test_batch):

            noise_shape = (args.batch_size,block_len, coderate_n)
            if mod_type != "LDPC" or mod_type != "POLAR":
                mod = get_modem( mod_type)
                X_test = torch.randint(0, mod.M, (args.batch_size, block_len, coderate_k), dtype=torch.float)
            else:
                X_test = torch.randint(0, 2, (args.batch_size, block_len, coderate_k), dtype=torch.float)

            # train encoder/decoder with different SNR... seems to be a good practice.
            SNR = args.train_enc_channel_low
            code_rate = coderate_k/ coderate_n
            test_bce_loss, test_custom_loss, test_ber, test_bler = \
                validate_noise(model, optimizer,X_test, args,SNR,noise_shape,use_cuda,"AWGN",coderate_k,coderate_n,mod_type)
            test_bce_loss_awgn +=test_bce_loss
            test_custom_loss_awgn +=test_custom_loss
            test_ber_awgn +=test_ber
            test_bler_awgn += test_bler
            test_bce_loss, test_custom_loss, test_ber, test_bler = \
                validate_noise(model, optimizer, X_test, args, SNR, noise_shape,use_cuda,"Rayleigh", coderate_k,coderate_n,mod_type)
            test_bce_loss_ray += test_bce_loss
            test_custom_loss_ray += test_custom_loss
            test_ber_ray += test_ber
            test_bler_ray += test_bler
            test_bce_loss, test_custom_loss, test_ber, test_bler = \
                validate_noise(model, optimizer, X_test, args, SNR, noise_shape,use_cuda,"Rician", coderate_k,coderate_n,mod_type)
            test_bce_loss_rici += test_bce_loss
            test_custom_loss_rici += test_custom_loss
            test_ber_rici += test_ber
            test_bler_rici += test_bler
            
    test_bce_loss_awgn /= num_test_batch
    test_custom_loss_awgn /= num_test_batch
    test_ber_awgn  /= num_test_batch
    test_bler_awgn /= num_test_batch
    test_bce_loss_ray /= num_test_batch
    test_custom_loss_ray /= num_test_batch
    test_ber_ray /= num_test_batch
    test_bler_ray /= num_test_batch
    test_bce_loss_rici /= num_test_batch
    test_custom_loss_rici /= num_test_batch
    test_ber_rici /= num_test_batch
    test_bler_rici /= num_test_batch

    if verbose:
        print('====> Test set BCE loss for AWGN', float(test_bce_loss_awgn),
              'Custom Loss',float(test_custom_loss_awgn),
              'with ber ', float(test_ber_awgn),
              'with bler ', float(test_bler_awgn),
        )
        print('====> Test set BCE loss for Rayleigh', float(test_bce_loss_ray),
              'Custom Loss', float(test_custom_loss_ray),
              'with ber ', float(test_ber_ray),
              'with bler ', float(test_bler_ray),
              )
        print('====> Test set BCE loss for Rician', float(test_bce_loss_rici),
              'Custom Loss', float(test_custom_loss_rici),
              'with ber ', float(test_ber_rici),
              'with bler ', float(test_bler_rici),
              )
    report_loss = (float(test_bce_loss_awgn),float(test_bce_loss_ray),float(test_bce_loss_rici))
    report_ber  = (float(test_ber_awgn),float(test_ber_ray),float(test_ber_rici))
    report_bler = (float(test_bler_awgn),float(test_bler_ray),float(test_bler_rici))

    return report_loss, report_ber, report_bler

def test_noise(model, X_test, args,SNR,noise_shape,batch_idx, use_cuda,noise_type,coderate_k=1,coderate_n=3,mod_type="QAM16"):
    device = torch.device("cuda" if use_cuda else "cpu")
    code_rate = coderate_k / coderate_n
    fwd_noise, encoded_input, input_msg ,sim_ber = generate_noise_SNR_Sim(SNR, noise_shape,args,noise_type ,coderate_k,coderate_n,mod_type)
    noise = fwd_noise.to(device)
    
    fwd_noise = noise.real.float()
    X_test, fwd_noise = X_test.to(device), fwd_noise.to(device)
    X_hat_test, the_codes = model(X_test, fwd_noise)

    test_ber = errors_ber(X_hat_test, X_test)
    test_bler = errors_bler(X_hat_test, X_test)

    if batch_idx == 0:
        test_pos_ber = errors_ber_pos(X_hat_test, X_test)
        codes_power = code_power(the_codes)
    else:
        test_pos_ber = errors_ber_pos(X_hat_test, X_test)
        codes_power = code_power(the_codes)
        
    return test_ber,test_bler,codes_power,test_pos_ber,sim_ber

def test(model, filename, args, block_len=10, coderate_k=1, coderate_n=3,mod_type="QAM16", use_cuda=False):
    device = torch.device("cuda" if use_cuda else "cpu")
    model.eval()

    # Precomputes Norm Statistics.
    if args.precompute_norm_stats:
        with torch.no_grad():
            num_test_batch = int(args.num_block/(args.batch_size)* args.test_ratio)
            for batch_idx in range(num_test_batch):
                X_test = torch.randint(0, 2, (args.batch_size,block_len, coderate_k), dtype=torch.float)
                X_test = X_test.to(device)
                _      = model.enc(X_test)
            print('Pre-computed norm statistics mean ',model.enc.mean_scalar, 'std ', model.enc.std_scalar)

    LC_awgn_ber,LC_ray_ber,LC_rici_ber = [], [],[]
    Sim_awgn_ber, Sim_ray_ber, Sim_rici_ber = [], [], []

    # snr_interval = (args.snr_test_end - args.snr_test_start)* 1.0 /  (args.snr_points-1)
    # snrs = [snr_interval* item + args.snr_test_start for item in range(args.snr_points)]
    snrs = np.arange(0, 100, 5)

    print('SNRS', snrs)
    sigmas = snrs
    data_file = open(filename, 'a')
    for sigma, this_snr in zip(sigmas, snrs):
        LC_awgn_test_ber, LC_awgn_test_bler, LC_awgn_codes_power, LC_awgn_test_pos_ber = .0, .0, .0,.0
        LC_ray_test_ber, LC_ray_test_bler, LC_ray_codes_power ,LC_ray_test_pos_ber = .0, .0, .0, .0
        LC_rici_test_ber, LC_rici_test_bler, LC_rici_codes_power,LC_rici_test_pos_ber = .0, .0, .0, .0
        awgn_test_sim_ber,ray_test_sim_ber,rici_test_sim_ber =.0,.0,.0
        noise_shape = (args.batch_size, block_len, coderate_n)
        with torch.no_grad():
            num_test_batch = int(args.num_block/(args.batch_size))
            for batch_idx in range(num_test_batch):
                if mod_type != "LDPC" or mod_type != "POLAR":
                    mod = get_modem( mod_type)
                    X_test = torch.randint(0, mod.M, (args.batch_size, block_len, coderate_k), dtype=torch.float)

                else:
                    X_test = torch.randint(0, 2, (args.batch_size, block_len, coderate_k), dtype=torch.float)

                test_ber,test_bler,codes_power,test_pos_ber,test_sim_ber = test_noise(model, X_test, args,this_snr,noise_shape,batch_idx, use_cuda,"AWGN",coderate_k,coderate_n,mod_type)

                LC_awgn_test_ber += test_ber
                LC_awgn_test_bler += test_bler
                LC_awgn_codes_power += codes_power
                LC_awgn_test_pos_ber += test_pos_ber
                awgn_test_sim_ber += test_sim_ber
                test_ber, test_bler, codes_power, test_pos_ber,test_sim_ber = test_noise(model, X_test, args, this_snr, noise_shape,
                                                                            batch_idx, use_cuda, "Rayleigh", coderate_k,
                                                                            coderate_n,mod_type)
                LC_ray_test_ber += test_ber
                LC_ray_test_bler += test_bler
                LC_ray_codes_power += codes_power
                LC_ray_test_pos_ber += test_pos_ber
                ray_test_sim_ber += test_sim_ber
                test_ber, test_bler, codes_power, test_pos_ber,test_sim_ber = test_noise(model, X_test, args, this_snr, noise_shape,
                                                                            batch_idx, use_cuda, "Rician", coderate_k,
                                                                            coderate_n,mod_type)
                LC_rici_test_ber += test_ber
                LC_rici_test_bler += test_bler
                LC_rici_codes_power += codes_power
                LC_rici_test_pos_ber += test_pos_ber
                rici_test_sim_ber += test_sim_ber
                
            if args.print_pos_power:
                print('code power', codes_power/num_test_batch)
            if args.print_pos_ber:
                res_pos = test_pos_ber/num_test_batch
                res_pos_arg = np.array(res_pos.cpu()).argsort()[::-1]
                res_pos_arg = res_pos_arg.tolist()
                print('positional ber', res_pos)
                print('positional argmax',res_pos_arg)

        LC_awgn_test_ber  /= num_test_batch
        LC_awgn_test_bler /= num_test_batch
        LC_ray_test_ber  /= num_test_batch 
        LC_ray_test_bler  /= num_test_batch 
        LC_rici_test_ber /= num_test_batch 
        LC_rici_test_bler /= num_test_batch
        awgn_test_sim_ber /= num_test_batch
        ray_test_sim_ber /= num_test_batch
        rici_test_sim_ber /= num_test_batch


        print('Test SNR',this_snr ,'learn codes ber with awgn ', float(LC_awgn_test_ber),
              'learn codes ber with rayleigh ', float(LC_ray_test_ber),
              'learn codes ber with rician ', float(LC_rici_test_ber),
              'ber with awgn ', float(awgn_test_sim_ber),
              'ber with rayleigh ', float(ray_test_sim_ber),
              'ber with rician ', float(rici_test_sim_ber))
        data_file.write(str(block_len)+' '+str(coderate_k)+' '+str(coderate_n)+' '+str(sigma) + ' ' + str(float(LC_awgn_test_ber)) + ' ' +
                        str(float(LC_ray_test_ber)) + ' ' +str(float(LC_rici_test_ber))
                        + ' ' +str(float(awgn_test_sim_ber))+ ' ' +str(float(ray_test_sim_ber))+ ' ' +str(float(rici_test_sim_ber))+ "\n")

        LC_awgn_ber.append( float(LC_awgn_test_ber))
        LC_ray_ber.append(float(LC_ray_test_ber))
        LC_rici_ber.append(float(LC_rici_test_ber))
        Sim_awgn_ber.append(float(awgn_test_sim_ber))
        Sim_ray_ber.append(float(ray_test_sim_ber))
        Sim_rici_ber.append(float(rici_test_sim_ber))
    data_file.close()
    print('final results on SNRs ', snrs)
    print('Learn Codes AWGN', LC_awgn_ber)
    print('Learn Codes rayleigh', LC_ray_ber)
    print('Learn Codes rician', LC_rici_ber)
    print('AWGN', Sim_awgn_ber)
    print('rayleigh', Sim_ray_ber)
    print('rician', Sim_rici_ber)

    # # compute adjusted SNR. (some quantization might make power!=1.0)
    # enc_power = 0.0
    # with torch.no_grad():
    #     for idx in range(num_test_batch):
    #         X_test     = torch.randint(0, 2, (args.batch_size, block_len, coderate_k), dtype=torch.float)
    #         X_test     = X_test.to(device)
    #         X_code     = model.enc(X_test)
    #         enc_power +=  torch.std(X_code)
    # enc_power /= float(num_test_batch)
    # print('encoder power is',enc_power.item())
    # adj_snrs = [snr_sigma2db(snr_db2sigma(item)/enc_power) for item in snrs]
    # print('adjusted SNR should be',adj_snrs)