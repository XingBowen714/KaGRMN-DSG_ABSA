import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from model_gcn import GAT, Rel_GAT
from model_utils import GCN, MultiHeadAttention, LinearAttention, DotprodAttention, BiRelationAttention, RelationAttention, Highway, mask_logits
#from tree import *

class KaGRMN_DSG(nn.Module):
    
    def __init__(self, args, dep_tag_num, pos_tag_num):
        super(KaGRMN_DSG, self).__init__()
        self.args = args

        # Bert
        config = BertConfig.from_pretrained(args.bert_model_dir)
        self.bert = BertModel.from_pretrained(
            args.bert_model_dir, config=config, from_tf =False)
        self.des_bert = BertModel.from_pretrained(
            args.bert_model_dir, config=config, from_tf =False)
        self.dropout_bert = nn.Dropout(config.hidden_dropout_prob)
        self.dropout = nn.Dropout(args.dropout)
        args.embedding_dim = config.hidden_size  # 768

        if args.highway:
            self.highway_dep = Highway(args.num_layers, args.embedding_dim)
            self.highway = Highway(args.num_layers, args.embedding_dim)

        self.stack_num = args.stack_num

        self.norm_att = nn.LayerNorm(args.embedding_dim)
        self.norm_gcn = nn.LayerNorm(args.embedding_dim)
        self.norm_rat = nn.LayerNorm(args.embedding_dim)

        self.gcn = nn.ModuleList([GCN(args.embedding_dim, args.embedding_dim).to(args.device) for i in range(args.n_gcn)])

        # GAT
        self.gat_dep = [BiRelationAttention(in_dim=args.embedding_dim).to(args.device) for i in range(args.rel_num_heads)]

        self.t2d_att = DotprodAttention()
        self.degate_linear = nn.Linear(args.embedding_dim * 2, args.embedding_dim)
        self.degate_linear2 = nn.Linear(args.embedding_dim * 2, 1)

        self.self_attention = MultiHeadAttention(n_head = args.self_num_heads, d_model = args.embedding_dim, d_k = args.embedding_dim, d_v = args.embedding_dim)
        self.sf_linear = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.dep_embed = nn.Embedding(dep_tag_num, args.embedding_dim)
        #self.dep_embed2 = nn.Embedding(dep_tag_num, args.embedding_dim)
        
        self.fc_graph = nn.Linear(2 * args.embedding_dim, args.embedding_dim).to(args.device)
        last_hidden_size = args.embedding_dim * 2
        layers = [
            nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
        for _ in range(args.num_mlps - 1):
            layers += [nn.Linear(args.final_hidden_size,
                                 args.final_hidden_size), nn.ReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)

    def position_weight(self, x, asp_start, asp_end, text_len, aspect_len):
            batch_size = x.shape[0]
            seq_len = x.shape[1]
            asp_start = asp_start.int().cpu().numpy()
            asp_end = asp_end.int().cpu().numpy()
            text_len = text_len.int().cpu().numpy()
            aspect_len = aspect_len.int().cpu().numpy()
            text_len = text_len - aspect_len + 1
            weight = [[] for i in range(batch_size)]
            for i in range(batch_size):
                context_len = text_len[i] - 1
                for j in range(asp_start[i]):
                    weight[i].append(1-(asp_start[i]-j)/context_len)
                for j in range(asp_start[i], asp_start[i] + 1):
                    weight[i].append(1)
                for j in range(asp_start[i]+1, text_len[i]):
                    weight[i].append(1-(j-asp_start[i])/context_len)
                for j in range(text_len[i], seq_len):
                    weight[i].append(0)
            weight = torch.tensor(weight).unsqueeze(2).to(self.args.device)
            #print(weight)
            #print(asp_start, aspect_len, text_len)
            #assert 1==0
            return weight*x
        
    def forward(self, input_ids, input_aspect_ids, word_indexer, w_idx, aspect_indexer,input_cat_ids, \
        segment_ids, input_des_ids, des_indexer, pos_class, dep_tags, text_len, aspect_len, des_len, dep_rels, dep_heads, \
        aspect_position, dep_dirs, aspect_start, sparse_graph):

        ct_mask = (torch.ones_like(w_idx) != w_idx).float()  # (N，L)
        ct_mask[:,0] = 1
        aspect_end = aspect_start + aspect_len - 1 
        asp_mask = torch.zeros_like(dep_tags).float()
        for i in range(aspect_start.size()[0]):
            asp_mask[i][aspect_start[i] - 1] = 1
        #print(des_indexer)
        des_mask = (torch.ones_like(des_indexer) != des_indexer).float()  # (N，L)
        des_mask[:,0] = 1
        outputs = self.bert(input_cat_ids, token_type_ids = segment_ids)
        des_outputs = self.des_bert(input_des_ids)

        feature_output = outputs[0] # (N, L, D)
        pool_out = outputs[1] #(N, D)

        h_des = des_outputs[0]

        # index select, back to original batched size.
        c_h = torch.stack([torch.index_select(f, 0, w_i)
                               for f, w_i in zip(feature_output, w_idx)])
        h_des = torch.stack([torch.index_select(f, 0, w_i.long())
                               for f, w_i in zip(h_des, des_indexer)])

        # get the initial target representation which is the average pool of target words hidden states

        t_r = pool_out

        # target hidden states collapse, get ct_h_0
        
        for i in range(feature_output.size()[0]):
            
            c_h[i][aspect_start[i]-1] = t_r[i]

        # N x Stacks

        for i in range(self.stack_num):

            # target-description attention
            r_d = self.t2d_att(h_des, t_r, des_mask)
        
            # description embedding gate mechanism
            t_r = t_r +  torch.mul(r_d, self.degate_linear(torch.cat([t_r,r_d], dim = -1)))

            # space fitting and insert,  get ct_h_new
            t_r = self.sf_linear(t_r)

            for e in range(c_h.size()[0]):
                c_h[e][aspect_start[e]-1] = t_r[e]


            # self attention  

            c_h_st, att = self.self_attention(c_h, c_h, c_h, ct_mask)
            c_h = self.norm_att(c_h + c_h_st)
            t_r = torch.sum(asp_mask.unsqueeze(2) * c_h, dim = 1)
            # GCN layer
        dep_sparse_out_previous = c_h
        for i in range(self.args.n_gcn):
            dep_sparse_out = self.gcn[i](dep_sparse_out_previous, sparse_graph)
            dep_sparse_out = self.norm_gcn(dep_sparse_out_previous + dep_sparse_out)
            dep_sparse_out_previous = dep_sparse_out
            # Relational Graph Attention Layer
            
        dep_feature = self.dep_embed(dep_tags)
            
        if self.args.highway:
            dep_feature = self.highway_dep(dep_feature)

        dep_out = [g(c_h, dep_feature, ct_mask).unsqueeze(2) for g in self.gat_dep]  # (N, L, 1, D) * num_heads
        dep_out = torch.cat(dep_out, dim=2)  # (N, l, H, D)
        dep_dense_out = dep_out.mean(dim=2)  # (N, L, D)

        dep_dense_out = self.norm_rat(c_h + dep_dense_out)
            

            # concatenate and project
        c_h_cat = torch.cat((dep_sparse_out, dep_dense_out), dim = 2)
            
        c_h = self.fc_graph(c_h_cat)
           
            
        t_r = torch.sum(asp_mask.unsqueeze(2) * c_h, dim = 1)
       


        # res connect. contain the original information, more robust, smooth. balance the noisy information and useful information


        # classification

        t_r = t_r +  torch.mul(r_d, self.degate_linear2(torch.cat([t_r,r_d], dim = -1)))
        senti_feature = torch.cat([t_r,  pool_out], dim=1)  # (N, D')
         
        #print('t_r:{0}'.format(t_r))
        #assert 1==0
        x = self.dropout(senti_feature)
        x = self.fcs(x)
        logit = self.fc_final(x)
        #print('logit:{0}'.format(logit))
        #assert 1==0
        return logit


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0
