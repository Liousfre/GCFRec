import torch.nn as nn
import torch
import torch.nn.functional as F
import os
from diffurec import DiffuRec
from adrec import AdRec
from gcfrec import GCFRec
from dreamrec import DreamRec
from common import LayerNorm


class Att_Diffuse_model(nn.Module):
    def __init__(self, args):
        super(Att_Diffuse_model, self).__init__()
        self.emb_dim = args.hidden_size
        self.args = args
        self.item_num = args.item_num
        self.item_embedding = self.embed_item(pretrained=args.pretrained)
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.hist_norm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)
        self.diffu = create_model_diffu(args)
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=0)
        self.geodesic = args.geodesic

    def load_pretrained_emb_weight(self):
        path = os.path.join('saved', 'pretrain', self.args.dataset, 'pretrain.pth')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        return saved['item_embedding.weight']

    def embed_item(self, pretrained=False):
        if pretrained:
            weight = self.load_pretrained_emb_weight()
            return nn.Embedding.from_pretrained(weight, padding_idx=0, freeze=self.args.freeze_emb)
        return nn.Embedding(self.item_num + 1, self.emb_dim, padding_idx=0)

    def loss_rec(self, scores, labels):
        return self.loss_ce(scores, labels.squeeze(-1))

    def loss_diffu(self, rep_diffu, labels):
        scores = torch.matmul(rep_diffu, self.get_item_weight().t())
        scores_pos = scores.gather(1, labels)
        scores_neg_mean = (torch.sum(scores, dim=-1).unsqueeze(-1) - scores_pos) / (scores.shape[1] - 1)
        loss = torch.min(-torch.log(torch.mean(torch.sigmoid((scores_pos - scores_neg_mean).squeeze(-1)))), torch.tensor(1e8))
        return loss

    def get_item_weight(self):
        return self.item_embedding.weight

    def calculate_loss(self, out_seq, labels):
        index = labels > 0
        out_seq_flat = out_seq[index]
        labels_flat = labels[index]
        scores = torch.matmul(out_seq_flat, self.get_item_weight().t())
        loss = self.loss_ce(scores.reshape(-1, scores.shape[-1]), labels_flat.reshape(-1))
        return loss

    def calculate_score(self, item):
        scores = torch.matmul(item.reshape(-1, item.shape[-1]), self.get_item_weight().t())
        return scores

    def forward(self, sequence, tag, train_flag=True):
        item_embeddings = self.item_embedding(sequence)
        tag_embeddings = self.item_embedding(tag)
        if self.geodesic:
            tag_embeddings = F.normalize(tag_embeddings, p=2, dim=-1)
        item_embeddings = self.embed_dropout(item_embeddings)
        item_embeddings = self.hist_norm(item_embeddings)

        mask_seq = (sequence > 0).float()
        mask_tag = (tag > 0).float().view(tag.shape[0], -1)

        if isinstance(self.diffu, GCFRec):
            self.diffu.set_item_ids(sequence)

        if train_flag:
            out_seq, dif_loss = self.diffu(item_embeddings, tag_embeddings, mask_seq, mask_tag)
            last_item = out_seq[:, -1, :]
        else:
            out_seq = self.diffu.denoise_sample(item_embeddings, tag_embeddings, mask_seq, mask_tag)
            last_item = out_seq[:, -1, :]
            dif_loss = None
        return out_seq, last_item, dif_loss


def create_model_diffu(args):
    if args.model == 'diffurec':
        return DiffuRec(args)
    elif args.model == 'adrec':
        return AdRec(args)
    elif args.model == 'gcfrec':
        return GCFRec(args)
    elif args.model == 'dreamrec':
        return DreamRec(args)
    else:
        print('args.model is wrong')
        return None
