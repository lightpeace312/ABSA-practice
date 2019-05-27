import torch
import torch.nn as nn
from torch.autograd import Variable
from layers.attention import Attention, ARAttention
from layers.point_wise_feed_forward import PositionwiseFeedForward

class AttentionRNNCell(nn.Module):
    #这里自己实现一个RNN.
   #input_size就是char_vacab_size=26,hidden_size随意，就是隐层神经元数，output_size要分成categories类
    def __init__(self, n_heads, input_dim, attention_hidden_dim, hidden_dim, output_dim, inner_heads = 1, score_function='mlp', dropout = 0.1):
        super(AttentionRNNCell, self).__init__()

        self.hidden_dim = hidden_dim

        self.attn = ARAttention(input_dim, 
                        activation_dim = hidden_dim, 
                        out_dim = attention_hidden_dim, 
                        n_heads=inner_heads,
                        score_function=score_function, 
                        dropout=0.1)
                        
        self.pw_ffn = PositionwiseFeedForward(input_dim + attention_hidden_dim, d_inner_hid = hidden_dim, dropout = dropout)
        self.out = nn.Linear(hidden_dim*n_heads, output_dim)
        self.softmax = nn.LogSoftmax()

    def forward(self, inputs, hidden):
        attn_out = self.attn(inputs, hidden)
        concat = torch.cat((inputs, attn_out), 1)
        
        hidden = self.pw_ffn(concat) #N*hidden_size，这里计算了一个hidden，hidden会带来下一个combined里
        concat_hid = torch.cat(hidden)
        output = self.out(concat_hid) # N*output_size,就是一个普通全连接层
        output = self.softmax(output)#softmax
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))#hidden=[1,hidden_size]


class AttentionRNN(nn.Module):
    #这里自己实现一个RNN.
   #input_size就是char_vacab_size=26,hidden_size随意，就是隐层神经元数，output_size要分成categories类
    def __init__(self, n_heads, input_dim, hidden_dim, attention_hidden_dim, output_dim, score_function = 'mlp',return_sequence = False):
        super(AttentionRNN, self).__init__()

        cell = AttentionRNNCell(n_heads = n_heads,
                                input_dim = input_dim,
                                attention_hidden_dim = attention_hidden_dim,
                                hidden_dim = hidden_dim,
                                output_dim = output_dim,
                                score_function = score_function)

    def forward(self, inputs, hidden):
        bs, seq_len, _ = inputs.size()
        hidden = torch.zeros(self.hidden_dim).to(inputs.device)
        output_seq = []
        for i in range(seq_len):
            output, hidden = self.cell(inputs[:, i, :],hidden)
            output_seq.append(output.unsqueeze(bs))
        
        if self.return_sequence:
            return output_seq, hidden
        else:
            return output, hidden

    # def initHidden(self):
    #     return Variable(torch.zeros(1, self.hidden_size))#hidden=[1,hidden_size]

n_hidden = 128
target_size = 10#分类问题，分成几类
rnn = RNN(26, n_hidden, target_size)
hidden = rnn.initHidden()
#如果输入input，是一个char，按one-hot编码的，假设char空间是26（小写英文字母）
input = Variable(torch.zeros((1,26)))#[torch.FloatTensor of size 1x26],成员都是0
print(input)
output, next_hidden = rnn(input, hidden)#得到[1*10,1*128]
print(output.data.size(),next_hidden.data.size())