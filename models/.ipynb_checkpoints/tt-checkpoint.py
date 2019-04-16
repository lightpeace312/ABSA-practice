# -*- coding: utf-8 -*-
# file: aen.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
from layers.squeeze_embedding import SqueezeEmbedding
from layers.attention import Attention, NoQueryAttention, BearAttention
from layers.point_wise_feed_forward import PositionwiseFeedForward
import torch
import torch.nn as nn
import torch.nn.functional as F

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
# CrossEntropyLoss for Label Smoothing Regularization
class CrossEntropyLoss_LSR(nn.Module):
    def __init__(self, device, para_LSR=0.2):
        super(CrossEntropyLoss_LSR, self).__init__()
        self.para_LSR = para_LSR
        self.device = device
        self.logSoftmax = nn.LogSoftmax(dim=-1)

    def _toOneHot_smooth(self, label, batchsize, classes):
        prob = self.para_LSR * 1.0 / classes
        one_hot_label = torch.zeros(batchsize, classes) + prob
        for i in range(batchsize):
            index = label[i]
            one_hot_label[i, index] += (1.0 - self.para_LSR)
        return one_hot_label

    def forward(self, pre, label, size_average=True):
        b, c = pre.size()
        one_hot_label = self._toOneHot_smooth(label, b, c).to(self.device)
        loss = torch.sum(-one_hot_label * self.logSoftmax(pre), dim=1)
        if size_average:
            return torch.mean(loss)
        else:
            return torch.sum(loss)


class TargetedTransformer(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(TargetedTransformer, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding()
        
#         self.position_enc = nn.Embedding.from_pretrained(
#             get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
#             freeze=True)
#         self.attn_text = BearAttention(
#             opt.embed_dim,
#             out_dim=opt.hidden_dim,
#             n_head=8,
#             score_function='dot_product',
#             dropout=opt.dropout)
        self.attn_text = Attention(
            opt.embed_dim,
            out_dim=opt.hidden_dim,
            n_head=8,
            score_function='dot_product',
            dropout=opt.dropout)
        
        self.attn_text2 = Attention(
            opt.hidden_dim,
            out_dim=opt.hidden_dim,
            n_head=8,
            score_function='dot_product',
            dropout=opt.dropout)

    
        self.attn_aspect = Attention(
            opt.embed_dim,
            out_dim=opt.hidden_dim,
            n_head=8,
            score_function='dot_product',
            dropout=opt.dropout)
        
#         self.attn_aspect2 = BearAttention(
#             opt.hidden_dim,
#             out_dim=opt.hidden_dim,
#             n_head=8,
#             score_function='dot_product',
#             dropout=opt.dropout)
        
        self.ffn_c = PositionwiseFeedForward(
            opt.hidden_dim, 
            dropout=opt.dropout)
        
        self.ffn_c2 = PositionwiseFeedForward(
            opt.hidden_dim, 
            dropout=opt.dropout)

        self.ffn_t = PositionwiseFeedForward(
            opt.hidden_dim, 
            dropout=opt.dropout)
        
#         self.ffn_t2 = PositionwiseFeedForward(
#             opt.hidden_dim, 
#             dropout=opt.dropout)
        
        self.attn_s1 = Attention(
            opt.hidden_dim,
            n_head=8,
            score_function='dot_product',
            dropout=opt.dropout)
#         self.layer_norm1 = nn.LayerNorm(opt.hidden_dim)
#         self.layer_norm2 = nn.LayerNorm(opt.hidden_dim)

#         self.lstm =  DynamicLSTM(
#             opt.hidden_dim,
#             opt.hidden_dim,
#             num_layers=1,
#             only_use_last_hidden_state=True,
#             batch_first=True)
#         self.lstm = nn.LSTM(
#                 opt.hidden_dim, opt.hidden_dim, num_layers=1,
#                 bias=True, batch_first=True, dropout=opt.dropout, bidirectional=False)
        self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices, target_indices = inputs[0], inputs[1]
        context_len = torch.sum(text_raw_indices != 0, dim=-1)
        target_len = torch.sum(target_indices != 0, dim=-1)
        context = self.embed(text_raw_indices)
        context = self.squeeze_embedding(context, context_len)
#         enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)
        target = self.embed(target_indices)
        target = self.squeeze_embedding(target, target_len)
        
#         resdual1 = context
        hc, _ = self.attn_text(context, context)
        print(hc.size())
        hc = self.ffn_c(hc)
#         hc  = self.layer_norm1(hc)
#         resdual2 = hc
        hc, _ = self.attn_text2(hc, hc)
        hc = self.ffn_c2(hc)
#         hc  = self.layer_norm1(hc+resdual2)
        ht, _ = self.attn_aspect(target, target)
        ht = self.ffn_t(ht)
        
#         ht, _ = self.attn_aspect2(ht, ht)
#         ht = self.ffn_t2(ht)
        
        s1, _ = self.attn_s1(hc, ht) #(?,300,t)

        context_len = torch.tensor(
            context_len, dtype=torch.float).to(self.opt.device)
            
        target_len = torch.tensor(
            target_len, dtype=torch.float).to(self.opt.device)

        
        s1_mean = torch.div(
            torch.sum(s1, dim=1), context_len.view(context_len.size(0), 1))
#         x = self.lstm(s1, target_len)
       
        out = self.dense(s1_mean)
        return out


class AEN_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(AEN_BERT, self).__init__()
        self.opt = opt
        self.bert = bert
        self.squeeze_embedding = SqueezeEmbedding()
        self.dropout = nn.Dropout(opt.dropout)

        self.attn_k = Attention(opt.bert_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        self.attn_q = Attention(opt.bert_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        self.ffn_c = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)
        self.ffn_t = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)

        self.attn_s1 = Attention(opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)

        self.dense = nn.Linear(opt.hidden_dim*3, opt.polarities_dim)

    def forward(self, inputs):
        context, target = inputs[0], inputs[1]
        context_len = torch.sum(context != 0, dim=-1)
        target_len = torch.sum(target != 0, dim=-1)
        context = self.squeeze_embedding(context, context_len)
        context, _ = self.bert(context, output_all_encoded_layers=False)
        context = self.dropout(context)
        target = self.squeeze_embedding(target, target_len)
        target, _ = self.bert(target, output_all_encoded_layers=False)
        target = self.dropout(target)

        hc, _ = self.attn_k(context, context)
        hc = self.ffn_c(hc)
        ht, _ = self.attn_q(context, target)
        ht = self.ffn_t(ht)

        s1, _ = self.attn_s1(hc, ht)

        context_len = torch.tensor(context_len, dtype=torch.float).to(self.opt.device)
        target_len = torch.tensor(target_len, dtype=torch.float).to(self.opt.device)

        hc_mean = torch.div(torch.sum(hc, dim=1), context_len.view(context_len.size(0), 1))
        ht_mean = torch.div(torch.sum(ht, dim=1), target_len.view(target_len.size(0), 1))
        s1_mean = torch.div(torch.sum(s1, dim=1), context_len.view(context_len.size(0), 1))

        x = torch.cat((hc_mean, s1_mean, ht_mean), dim=-1)
        out = self.dense(x)
        return out

