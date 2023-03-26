import torch.nn.functional as F
import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(2*dec_hid_dim, 1, bias=False)
        
    def forward(self, s, enc_output):
        
        batch_size = enc_output.shape[0]
        src_len = enc_output.shape[1]

        s = s.repeat(1, src_len, 1)
        attention = self.attn(torch.cat((s, enc_output), dim = 2)).transpose(1,2)
        
        return F.softmax(attention, dim=2)


class DEC(torch.nn.Module):
    def __init__(self, args,block_len=10,coderate_k=1,coderate_n=3):
        super(DEC, self).__init__()
        self.args = args
        self.block_len = block_len
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        if args.dec_rnn == 'gru':
            RNN_MODEL = torch.nn.GRU
        elif args.dec_rnn == 'lstm':
            RNN_MODEL = torch.nn.LSTM
        else:
            RNN_MODEL = torch.nn.RNN

        self.dropout = torch.nn.Dropout(args.dropout)

        # attention
        self.attention = Attention(args.enc_num_unit, args.dec_num_unit)
        self.fc = torch.nn.Linear(2*args.dec_num_unit, args.dec_num_unit)

        self.dec1_rnns = RNN_MODEL(int(coderate_n),  args.dec_num_unit,
                                                    num_layers=args.dec_num_layer, bias=True, batch_first=True,
                                                    dropout=args.dropout)

        self.dec2_rnns = RNN_MODEL(int(coderate_n),  args.dec_num_unit,
                                        num_layers=args.dec_num_layer, bias=True, batch_first=True,
                                        dropout=args.dropout)

        self.dec_outputs = torch.nn.Linear(2*args.dec_num_unit, coderate_k)

    
    def dec_act(self, inputs):
        if self.args.dec_act == 'tanh':
            return  F.tanh(inputs)
        elif self.args.dec_act == 'elu':
            return F.elu(inputs)
        elif self.args.dec_act == 'relu':
            return F.relu(inputs)
        elif self.args.dec_act == 'selu':
            return F.selu(inputs)
        elif self.args.dec_act == 'sigmoid':
            return F.sigmoid(inputs)
        elif self.args.dec_act == 'linear':
            return inputs
        else:
            return inputs

    def forward(self, received):
        received = received.type(torch.FloatTensor).to(self.this_device)
        out1,_ = self.dec1_rnns(received)
        out2,_ = self.dec2_rnns(received)

        rnn_out1 = torch.zeros([out1.shape[0],out1.shape[1],out1.shape[2]]).type(torch.FloatTensor).to(self.this_device)
        rnn_out2 = torch.zeros([out2.shape[0],out2.shape[1],out2.shape[2]]).type(torch.FloatTensor).to(self.this_device)
        rnn_out1[:,0,:] = out1[:,0,:]
        rnn_out2[:,0,:] = out2[:,0,:]

        for i in range(self.block_len):
            if i>0 and i<5:
                rnn_out1[:,i,:] = out1[:,i,:]
                rnn_out2[:,i,:] = out2[:,i,:]
            if i>=5:
                a1 = self.attention(out1[:,i:i+1,:], out1[:,i-5:i,:])
                a2 = self.attention(out2[:,i:i+1,:], out2[:,i-5:i,:])

                c1 = torch.bmm(a1, out1[:,i-5:i,:])
                c2 = torch.bmm(a2, out2[:,i-5:i,:])

                ith_out1 = self.fc(torch.cat((c1,out1[:,i:i+1,:]),dim=2)).squeeze(1)
                ith_out2 = self.fc(torch.cat((c2,out2[:,i:i+1,:]),dim=2)).squeeze(1)

                rnn_out1[:,i,:] = ith_out1
                rnn_out2[:,i,:] = ith_out2

        for i in range(self.block_len):
            if (i>=self.block_len-self.args.D-1):
                rt_d = rnn_out2[:,self.block_len-1:self.block_len,:]
            else:
                rt_d = rnn_out2[:,i+self.args.D:i+self.args.D+1,:]
            rt = rnn_out1[:,i:i+1,:]
            rnn_out = torch.cat((rt, rt_d), dim=2)
            dec_out = self.dec_act(self.dec_outputs(rnn_out))
            if i==0:
                final = dec_out
            else:
                final = torch.cat((final,dec_out),dim=1)
        final = torch.sigmoid(final)

        return final