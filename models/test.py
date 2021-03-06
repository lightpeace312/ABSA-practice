from layers.dynamic_rnn import DynamicLSTM
from layers.squeeze_embedding import SqueezeEmbedding
from layers.attention import Attention, NoQueryAttention
from layers.point_wise_feed_forward import PositionwiseFeedForward
import torch

import torch.nn as nn
import torch.nn.functional as F
from layers.high_layers import EncoderLayer, DecoderLayer


# # CrossEntropyLoss for Label Smoothing Regularization
# class CrossEntropyLoss_LSR(nn.Module):
#     def __init__(self, device, para_LSR=0.2):
#         super(CrossEntropyLoss_LSR, self).__init__()
#         self.para_LSR = para_LSR
#         self.device = device
#         self.logSoftmax = nn.LogSoftmax(dim=-1)

#     def _toOneHot_smooth(self, label, batchsize, classes):
#         prob = self.para_LSR * 1.0 / classes
#         one_hot_label = torch.zeros(batchsize, classes) + prob
#         for i in range(batchsize):
#             index = label[i]
#             one_hot_label[i, index] += (1.0 - self.para_LSR)
#         return one_hot_label

#     def forward(self, pre, label, size_average=True):
#         b, c = pre.size()
#         one_hot_label = self._toOneHot_smooth(label, b, c).to(self.device)
#         loss = torch.sum(-one_hot_label * self.logSoftmax(pre), dim=1)
#         if size_average:
#             return torch.mean(loss)
#         else:
#             return torch.sum(loss)


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_tgt_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()
        # n_position = len_max_seq + 1

        # self.tgt_word_emb = nn.Embedding(
        #     n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)

        # self.position_enc = nn.Embedding.from_pretrained(
        #     get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
        #     freeze=True)

        # self.layer_stack = nn.ModuleList([
        #     DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
        #     for _ in range(n_layers)])
        self.decoderlayer =  DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

class Test(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(Test, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding()

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=opt.dropout) 
            for _ in range(n_layers)])
    
        # self.ffn_c = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)
        # self.ffn_t = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)

        self.attn_s1 = Attention(opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)

        self.dense = nn.Linear(opt.hidden_dim*3, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices, target_indices = inputs[0], inputs[1]
        context_len = torch.sum(text_raw_indices != 0, dim=-1)
        target_len = torch.sum(target_indices != 0, dim=-1)
        context = self.embed(text_raw_indices)
        context = self.squeeze_embedding(context, context_len)
        target = self.embed(target_indices)
        target = self.squeeze_embedding(target, target_len)

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        # hc, _ = self.attn_k(context, context)
        # hc = self.ffn_c(hc)
        # ht, _ = self.attn_q(context, target)
        # ht = self.ffn_t(ht)

        # s1, _ = self.attn_s1(hc, ht)

        # context_len = torch.tensor(context_len, dtype=torch.float).to(self.opt.device)
        # target_len = torch.tensor(target_len, dtype=torch.float).to(self.opt.device)

        # hc_mean = torch.div(torch.sum(hc, dim=1), context_len.view(context_len.size(0), 1))
        # ht_mean = torch.div(torch.sum(ht, dim=1), target_len.view(target_len.size(0), 1))
        # s1_mean = torch.div(torch.sum(s1, dim=1), context_len.view(context_len.size(0), 1))

        # x = torch.cat((hc_mean, s1_mean, ht_mean), dim=-1)
        out = self.dense(dec_output)
        return out