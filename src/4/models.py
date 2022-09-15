import torch
from torch import nn
import torch.nn.functional as F
from modules.transformer import TransformerEncoder

class HUSFORMERModel(nn.Module):
    def __init__(self, hyp_params):
        super(HUSFORMERModel, self).__init__()
        self.orig_d_m1, self.orig_d_m2, self.orig_d_m3,self.orig_d_m4  = hyp_params.orig_d_m1, hyp_params.orig_d_m2, hyp_params.orig_d_m3,hyp_params.orig_d_m4
        self.d_m = 30
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask

        combined_dim = 30     
        output_dim = hyp_params.output_dim        
        self.channels = hyp_params.m1_len+hyp_params.m2_len+hyp_params.m3_len+hyp_params.m4_len
        
        # 1. Temporal convolutional layers
        self.proj_m1 = nn.Conv1d(self.orig_d_m1, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m2 = nn.Conv1d(self.orig_d_m2, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m3 = nn.Conv1d(self.orig_d_m3, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m4 = nn.Conv1d(self.orig_d_m4, self.d_m, kernel_size=1, padding=0, bias=False)
        self.final_conv = nn.Conv1d(self.channels, 1, kernel_size=1, padding=0, bias=False)
        
        # 2. Cross-modal Attentions
        self.trans_m1_all = self.get_network(self_type='m1_all', layers=3)
        self.trans_m2_all = self.get_network(self_type='m2_all', layers=3)
        self.trans_m3_all = self.get_network(self_type='m3_all', layers=3)
        self.trans_m4_all = self.get_network(self_type='m4_all', layers=3)
        
        # 3. Self Attentions 
        self.trans_final = self.get_network(self_type='policy', layers=5)
        
        # 4. Projection layers
        self.proj1 = self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['m1_all','m2_all','m3_all','m4_all','policy']:
            embed_dim, attn_dropout = self.d_m, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
            
    def forward(self,m1,m2,m3,m4):

        m_1 = m1.transpose(1, 2)
        m_2 = m2.transpose(1, 2)
        m_3 = m3.transpose(1, 2)
        m_4 = m4.transpose(1, 2)
        # Project features
        proj_x_m1 = m_1 if self.orig_d_m1 == self.d_m else self.proj_m1(m_1)
        proj_x_m2 = m_2 if self.orig_d_m2 == self.d_m else self.proj_m2(m_2)
        proj_x_m3 = m_3 if self.orig_d_m3 == self.d_m else self.proj_m3(m_3)
        proj_x_m4 = m_4 if self.orig_d_m4 == self.d_m else self.proj_m4(m_4)

        proj_x_m1 = proj_x_m1.permute(2, 0, 1)
        proj_x_m2 = proj_x_m2.permute(2, 0, 1)
        proj_x_m3 = proj_x_m3.permute(2, 0, 1)
        proj_x_m4 = proj_x_m4.permute(2, 0, 1)
        
        proj_all = torch.cat([proj_x_m1 , proj_x_m2 , proj_x_m3 , proj_x_m4], dim=0)
            
        m1_with_all = self.trans_m1_all(proj_x_m1, proj_all, proj_all)  
        m2_with_all = self.trans_m2_all(proj_x_m2, proj_all, proj_all)  
        m3_with_all = self.trans_m3_all(proj_x_m3, proj_all, proj_all)  
        m4_with_all = self.trans_m4_all(proj_x_m4, proj_all, proj_all)   

        last_hs1 = torch.cat([m1_with_all, m2_with_all, m3_with_all, m4_with_all] , dim = 0)
        last_hs2 = self.trans_final(last_hs1).permute(1, 0, 2)
        last_hs = self.final_conv(last_hs2).squeeze(1)

        output = self.out_layer(last_hs)

        return output, last_hs
