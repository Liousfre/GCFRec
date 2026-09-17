"""
GRU4Rec
################################################

Reference:
    Balazs Hidasi et al. "Session-based Recommendations with Recurrent
    Neural Networks." in ICLR 2016.
"""

import torch
from torch import nn


class GRU4Rec(nn.Module):
    def __init__(self, args):
        super(GRU4Rec, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.item_num = args.item_num
        self.num_layers = 2

        self.item_embedding = nn.Embedding(args.item_num + 1, self.hidden_size, padding_idx=0)
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=args.dropout if self.num_layers > 1 else 0.0,
        )
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
        item_emb = self.item_embedding(item_seq)
        return item_emb, None

    def forward(self, item_seq, tgt_seq, train_flag=True):
        item_emb = self.item_embedding(item_seq)
        item_emb = self.LayerNorm(item_emb)
        item_emb = self.dropout(item_emb)
        output_seq, _ = self.gru(item_emb)
        last_item = output_seq[:, -1, :]
        return output_seq, last_item

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
