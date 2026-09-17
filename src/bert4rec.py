"""
BERT4Rec
################################################

Reference:
    Fei Sun et al. "BERT4Rec: Sequential Recommendation with Bidirectional
    Encoder Representations from Transformer." in CIKM 2019.

"""

import torch
from torch import nn
from common import TransformerEncoder


class BERT4Rec(nn.Module):
    def __init__(self, args):
        super(BERT4Rec, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.item_num = args.item_num
        self.mask_token = args.item_num + 1
        self.mask_ratio = getattr(args, 'bert_mask_ratio', 0.2)

        self.item_embedding = nn.Embedding(args.item_num + 2, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(args.max_len, self.hidden_size)
        self.trm_encoder = TransformerEncoder(args, num_blocks=2, norm_first=False, is_causal=False)
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

    def mask_training_batch(self, item_seq):
        """Apply BERT4Rec's masked-item objective with 80/10/10 corruption."""
        valid = item_seq.gt(0)
        selected = valid & (torch.rand(item_seq.shape, device=item_seq.device) < self.mask_ratio)

        # Every non-empty sequence contributes at least one prediction target.
        missing = valid.any(dim=1) & ~selected.any(dim=1)
        if missing.any():
            rows = missing.nonzero(as_tuple=False).squeeze(1)
            for row in rows.tolist():
                positions = valid[row].nonzero(as_tuple=False).squeeze(1)
                chosen = positions[torch.randint(positions.numel(), (1,), device=item_seq.device)]
                selected[row, chosen] = True

        labels = torch.where(selected, item_seq, torch.zeros_like(item_seq))
        corrupted = item_seq.clone()
        replacement = torch.rand(item_seq.shape, device=item_seq.device)
        corrupted[selected & (replacement < 0.8)] = self.mask_token

        random_positions = selected & (replacement >= 0.8) & (replacement < 0.9)
        random_items = torch.randint(
            1, self.item_num + 1, item_seq.shape, device=item_seq.device
        )
        corrupted[random_positions] = random_items[random_positions]
        # The remaining 10% retain the original item.
        return corrupted, labels

    def prepare_inference_batch(self, item_seq):
        """Append [MASK] after the observed prefix for next-item prediction."""
        shifted = torch.zeros_like(item_seq)
        shifted[:, :-1] = item_seq[:, 1:]
        shifted[:, -1] = self.mask_token
        return shifted

    def encode_sequence(self, item_seq):
        """Encode an item sequence without applying next-item masking."""
        item_emb, position_emb = self.embedding_layer(item_seq)
        input_emb = item_emb + position_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        mask_seq = (item_seq > 0).float()
        output_seq = self.trm_encoder(input_emb, mask_seq)
        last_item = output_seq[:, -1, :]
        return output_seq, last_item

    def forward(self, item_seq, tgt_seq, train_flag=True):
        if not train_flag:
            item_seq = self.prepare_inference_batch(item_seq)
        return self.encode_sequence(item_seq)

    def calculate_loss(self, seq_output, tgt_seq):
        index = tgt_seq > 0
        seq_output = seq_output[index]
        tgt_seq = tgt_seq[index]
        logits = torch.matmul(seq_output, self.item_embedding.weight[:self.item_num + 1].t())
        loss = self.loss_fct(logits.reshape(-1, logits.shape[-1]), tgt_seq.reshape(-1))
        return loss

    def calculate_score(self, item):
        scores = torch.matmul(item.reshape(-1, item.shape[-1]), self.item_embedding.weight[:self.item_num + 1].t())
        return scores
