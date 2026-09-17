"""
EulerFormer
################################################

Reference:
    Zhen Tian et al. "EulerFormer: Sequential User Behavior Modeling
    with Complex Vector Attention." in SIGIR '24.
"""

import torch
from torch import nn
from common import EulerFormerBlock


class EulerFormer(nn.Module):
    def __init__(self, args):
        super(EulerFormer, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.item_num = args.item_num
        self.num_blocks = 2
        self.heads = 4

        self.item_embedding = nn.Embedding(args.item_num + 1, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(args.max_len, self.hidden_size)
        self.trm_encoder = nn.ModuleList([
            EulerFormerBlock(self.hidden_size, self.heads, args.dropout,
                             is_causal=True, norm_first=False)
            for _ in range(self.num_blocks)
        ])

        self.LayerNorm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(args.emb_dropout)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.1)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def embedding_layer(self, item_seq):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        ).unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        return item_emb, position_embedding

    def forward(self, item_seq, tgt_seq, train_flag=True):
        item_emb, _ = self.embedding_layer(item_seq)
        # EulerFormer 通过复数旋转隐式编码位置, 不再加 position_embedding
        input_emb = self.LayerNorm(item_emb)
        input_emb = self.dropout(input_emb)
        mask_seq = (item_seq > 0).float()
        hidden = input_emb
        for block in self.trm_encoder:
            hidden = block(hidden, mask_seq)
        last_item = hidden[:, -1, :]
        return hidden, last_item

    def calculate_loss(self, seq_output, tgt_seq):
        index = tgt_seq > 0
        seq_output = seq_output[index]
        tgt_seq = tgt_seq[index]
        logits = torch.matmul(seq_output, self.item_embedding.weight.t())
        loss = self.loss_fct(logits.reshape(-1, logits.shape[-1]), tgt_seq.reshape(-1))
        return loss

    def calculate_score(self, item):
        scores = torch.matmul(item.reshape(-1, item.shape[-1]), self.item_embedding.weight.t())
        return scores
