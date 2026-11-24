"""三地区融合EMAformer：独立区域嵌入/预测头、单份共享注意力骨干。"""
import torch
from torch import nn
from ._emaformer import EMAformer

ARCHITECTURE = 'three_region_emaformer_v1'


class JointEMAformer(nn.Module):
    def __init__(self, layout, parameters):
        super().__init__()
        self.layout, self.parameters_config = layout, parameters
        self.seq_len = parameters['seq_len']
        self.pred_len = layout['pred_len']
        self.input_size = layout['input_size']
        self.indices = layout['branch_indices']
        self.positions = [indexes.index(target) for indexes,target in
                          zip(self.indices,layout['target_indices'])]
        if parameters['d_model']%parameters['n_heads']:
            raise ValueError('隐藏维度必须整除注意力头数')
        self.branches = nn.ModuleList()
        for indexes,position in zip(self.indices,self.positions):
            branch = EMAformer(len(indexes),target_index=position,seq_len=self.seq_len,
                               pred_len=self.pred_len,d_model=parameters['d_model'],
                               n_heads=parameters['n_heads'],e_layers=parameters['e_layers'],
                               d_ff=parameters['d_model']*parameters['ff_multiplier'],
                               dropout=parameters['dropout'],
                               output_proj_dropout=parameters['head_dropout'],
                               cycle=max(1,parameters['cycle']),activation=parameters['activation'])
            branch.enc_embedding.dropout = nn.Dropout(parameters['embedding_dropout'])
            for layer in branch.encoder.attn_layers:
                layer.norm_first = parameters['norm_first']
                layer.attention.inner_attention.dropout = nn.Dropout(parameters['attention_dropout'])
            widths = [parameters['d_model']]
            for index in range(parameters['head_layers']-1):
                widths.append(parameters['d_model']*parameters['head_multiplier']**(index+1))
            widths.append(self.pred_len)
            head = []
            for index,(left,right) in enumerate(zip(widths,widths[1:])):
                head.append(nn.Linear(left,right))
                if index<len(widths)-2:
                    head.extend([nn.GELU() if parameters['activation']=='gelu' else nn.ReLU(),
                                 nn.Dropout(parameters['head_dropout'])])
            branch.projector = nn.Sequential(*head)
            if parameters['output_mode']=='delta':
                # 初始输出等于最后已知价格；训练学习未来价格的修正量。
                nn.init.zeros_(branch.projector[-1].weight)
                nn.init.zeros_(branch.projector[-1].bias)
            if parameters['cycle']==0:
                del branch.phase_embedding
                del branch.joint_embedding
            if not parameters['channel_embedding']:
                del branch.channel_embedding
            self.branches.append(branch)
        self.encoder = self.branches[0].encoder
        for branch in self.branches:
            del branch.encoder

    def forward(self, x, starts):
        if x.ndim!=3 or x.shape[1:]!=(self.seq_len,self.input_size):
            raise ValueError('历史输入形状与保存配置不一致')
        starts = torch.as_tensor(starts,device=x.device).long().reshape(-1)
        if len(starts)!=len(x):
            raise ValueError('周期位置数量不一致')
        p = self.parameters_config
        outputs = []
        for branch,indexes,position in zip(self.branches,self.indices,self.positions):
            values = x[:,:,indexes]
            if p['window_norm']:
                mean = values.mean(1,keepdim=True).detach()
                std = torch.sqrt(values.var(1,keepdim=True,unbiased=False)+1e-5)
            else:
                mean, std = torch.zeros_like(values[:,:1]), torch.ones_like(values[:,:1])
            embedded = branch.enc_embedding((values-mean)/std,None)
            if p['channel_embedding']:
                embedded = embedded+branch.channel_embedding[None]
            if p['cycle']:
                phase = starts%p['cycle']
                embedded = (embedded+branch.phase_embedding(phase)[:,None]+
                            branch.joint_embedding(phase).reshape(len(x),len(indexes),p['d_model']))
            encoded,_ = self.encoder(embedded,attn_mask=None)
            output = branch.projector((encoded+embedded)[:,position])
            if p['output_mode']=='delta':
                output = values[:,-1,position,None]+p['residual_scale']*output*std[:,0,position,None]
            else:
                output = output*std[:,0,position,None]+mean[:,0,position,None]
            outputs.append(output)
        return torch.stack(outputs,-1)
