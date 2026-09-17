"""
GCFRec: gated conditioning fusion for diffusion recommendation.
"""
import os
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import SiLU, TransformerEncoder
from sasrec import SASRec
from step_sample import *
from utils import _extract_into_tensor, exponential_mapping

class DenoisedModel(nn.Module):
    def __init__(self, args):
        super(DenoisedModel, self).__init__()
        self.hidden_size = args.hidden_size
        if args.dif_decoder =='mlp':
            self.decoder = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size * 4),
                                        SiLU(),
                                        nn.Linear(self.hidden_size * 4, self.hidden_size),
                                        nn.LayerNorm(self.hidden_size),
                                        )
        else:
            self.decoder = TransformerEncoder(args,num_blocks=2,norm_first=False,hidden_size=self.hidden_size)

        self.time_embed = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size * 4),
                                        SiLU(),
                                        nn.Linear(self.hidden_size * 4, self.hidden_size)
                                        )

        self.lambda_uncertainty = args.lambda_uncertainty


    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        assert dim % 2 == 0
        half = dim // 2
        freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half).to(device=timesteps.device)
        args = timesteps.unsqueeze(-1).float() * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward_cfg(self,c, x, t, mask_seq,mask_tgt,cfg_scale=1.0):
        cond_eps = self.forward(c,x, t,mask_seq,mask_tgt)
        uncond_eps = self.forward(c,x, t,mask_seq,mask_tgt,condition=False)
        eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        return eps


    def forward(self, rep_item, x_t, t, mask_seq,mask_tgt,condition=True):
        if condition is not True:  #CFG
            rep_item = torch.zeros_like(rep_item)
            # mask = torch.rand_like(mask_seq) > 0.5
            # rep_item = torch.where(mask.unsqueeze(-1), torch.zeros_like(rep_item), rep_item)
        t=t.reshape(x_t.shape[0],-1)
        time_emb = self.time_embed(self.timestep_embedding(t, self.hidden_size))
        lambda_uncertainty = self.lambda_uncertainty  ### fixed

        rep_diffu = rep_item + lambda_uncertainty * (x_t + time_emb)

        if isinstance(self.decoder, nn.Sequential):
            # 如果是 MLP，直接应用
            rep_diffu = self.decoder(rep_diffu)
        else:
            rep_diffu = self.decoder(rep_diffu, mask_seq)

        return rep_diffu

class FusionLayer(nn.Module):
    """Gated/projection fusion of two (B, L, D) sequences h_a and h_p."""
    def __init__(self, hidden_size, fusion_type='gate', gate_mode='token', n_heads=4, gate_init_bias=4.0):
        super().__init__()
        self.fusion_type = fusion_type
        self.gate_mode = gate_mode

        if fusion_type == 'sum':
            pass

        elif fusion_type == 'wsum':
            self.raw_alpha = nn.Parameter(torch.tensor([4.0]))

        elif fusion_type == 'concat':
            self.proj = nn.Linear(hidden_size * 2, hidden_size)
            with torch.no_grad():
                w = torch.zeros(hidden_size, hidden_size * 2)
                w[:, :hidden_size] = torch.eye(hidden_size)
                self.proj.weight.copy_(w)
                self.proj.bias.zero_()

        elif fusion_type == 'gate':
            out_dim = 1 if gate_mode == 'scalar' else hidden_size
            self.gate_net = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, out_dim),
                nn.Sigmoid(),
            )
            with torch.no_grad():
                self.gate_net[-2].bias.fill_(gate_init_bias)

        elif fusion_type == 'xattn':
            self.n_heads = n_heads
            self.q_proj = nn.Linear(hidden_size, hidden_size)
            self.k_proj = nn.Linear(hidden_size, hidden_size)
            self.v_proj = nn.Linear(hidden_size, hidden_size)
            self.out_proj = nn.Linear(hidden_size, hidden_size)
            self.res_alpha = nn.Parameter(torch.tensor([4.0]))
            with torch.no_grad():
                self.out_proj.weight.zero_()
                self.out_proj.bias.zero_()

        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(self, h_a, h_p):
        if self.fusion_type == 'sum':
            return h_a + h_p

        if self.fusion_type == 'wsum':
            a = torch.sigmoid(self.raw_alpha)
            return a * h_a + (1.0 - a) * h_p

        if self.fusion_type == 'concat':
            return self.proj(torch.cat([h_a, h_p], dim=-1))

        if self.fusion_type == 'gate':
            gate = self.gate_net(torch.cat([h_a, h_p], dim=-1))
            return gate * h_a + (1.0 - gate) * h_p

        if self.fusion_type == 'xattn':
            B, L, D = h_a.shape
            H = self.n_heads
            Dh = D // H
            q = self.q_proj(h_a).view(B, L, H, Dh).transpose(1, 2)
            k = self.k_proj(h_p).view(B, L, H, Dh).transpose(1, 2)
            v = self.v_proj(h_p).view(B, L, H, Dh).transpose(1, 2)
            att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
            att = F.softmax(att, dim=-1)
            ctx = torch.matmul(att, v).transpose(1, 2).contiguous().view(B, L, D)
            update = self.out_proj(ctx)
            a = torch.sigmoid(self.res_alpha)
            return a * h_a + (1.0 - a) * update


class GCFRec(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.hidden_size = args.hidden_size
        self.schedule_sampler_name = args.schedule_sampler_name
        self.diffusion_steps = args.diffusion_steps
        self.use_timesteps = space_timesteps(self.diffusion_steps, [self.diffusion_steps])
        betas = get_named_beta_schedule(args)
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        alphas = 1.0 - betas

        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        self.posterior_mean_coef1 = (betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod))
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

        self.num_timesteps = int(self.betas.shape[0])

        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_name, self.num_timesteps)
        self.timestep_map = self.time_map()
        self.rescale_timesteps = args.rescale_timesteps
        self.original_num_steps = len(betas)

        self.net = DenoisedModel(args)
        self.independent_diffusion = args.independent
        self.cfg_scale = args.cfg_scale
        self.geodesic = args.geodesic
        self.ag_encoder = TransformerEncoder(args, num_blocks=2, norm_first=False)

        self.phi = self._build_phi(args)
        self.fusion = FusionLayer(
            args.hidden_size,
            fusion_type=str(getattr(args, 'gcf_fusion_type', 'gate')),
            gate_mode=str(getattr(args, 'gcf_gate_mode', 'scalar')),
            gate_init_bias=float(getattr(args, 'gate_init_bias', 4.0)),
        )
        self._item_ids = None

    @staticmethod
    def _build_phi(args):
        phi_args = copy.copy(args)
        phi_args.model = 'pretrain'
        phi = SASRec(phi_args)
        ckpt_path = os.path.join('saved', 'pretrain', args.dataset, 'pretrain.pth')
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        phi.load_state_dict(state, strict=False)
        for p in phi.parameters():
            p.requires_grad_(False)
        phi.eval()
        return phi

    def set_item_ids(self, item_ids):
        self._item_ids = item_ids

    def encode_seq(self, item_rep, mask_seq):
        h_a = self.ag_encoder(item_rep, mask_seq)
        with torch.no_grad():
            h_p, _ = self.phi(self._item_ids, tgt_seq=None, train_flag=False)
        h = self.fusion(h_a, h_p)
        return h

    def q_sample(self, x_start, t, noise=None, mask=None):
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        if self.geodesic:
            x_start = F.normalize(x_start, p=2, dim=-1)
        x_t = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        if self.geodesic:
            x_t = exponential_mapping(x_start, x_t)
        if mask is None:
            return x_t
        mask = th.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)
        return th.where(mask == 0, x_start, x_t)

    def time_map(self):
        timestep_map = []
        for i in range(len(self.alphas_cumprod)):
            if i in self.use_timesteps:
                timestep_map.append(i)
        return timestep_map

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        assert (posterior_mean.shape[0] == x_start.shape[0])
        return posterior_mean

    def p_mean_variance(self, rep_item, x_t, t, mask_seq, mask_tag):
        if self.cfg_scale == 1.:
            x_0 = self.net(rep_item, x_t, self._scale_timesteps(t), mask_seq, mask_tag)
        else:
            x_0 = self.net.forward_cfg(rep_item, x_t, self._scale_timesteps(t), mask_seq, mask_tag, self.cfg_scale)
        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        model_log_variance = _extract_into_tensor(model_log_variance, t, x_t.shape)
        model_mean = self.q_posterior_mean_variance(x_start=x_0, x_t=x_t, t=t)
        return model_mean, model_log_variance

    def p_sample(self, item_rep, noise_x_t, t, mask_seq, mask_tag):
        model_mean, model_log_variance = self.p_mean_variance(item_rep, noise_x_t, t, mask_seq, mask_tag)
        noise = th.randn_like(noise_x_t)
        nonzero_mask = (t != 0).float().unsqueeze(-1)
        sample_xt = model_mean + nonzero_mask * th.exp(0.5 * model_log_variance) * noise
        if self.geodesic:
            sample_xt = F.normalize(sample_xt, p=2, dim=-1)
        return sample_xt

    def denoise_sample(self, seq, tgt, mask_seq, mask_tag):
        seq = self.encode_seq(seq, mask_seq)
        noise_x_t = th.randn_like(tgt)
        indices = list(range(self.num_timesteps))[::-1]
        for i in indices:
            t = th.tensor([0] * (seq.shape[1] - 1) + [i], device=seq.device).unsqueeze(0).repeat(seq.shape[0], 1)
            noise_x_t = torch.concat([tgt[:, :-1], noise_x_t[:, -1:]], dim=1)
            noise_x_t = self.p_sample(seq, noise_x_t, t, mask_seq, mask_tag)
        return noise_x_t

    def independent_diffuse(self, tgt, mask, is_independent=False):
        if is_independent:
            t, weights = self.schedule_sampler.sample(tgt.shape[0] * tgt.shape[1], tgt.device)
            t = t * mask.reshape(-1).long()
            x_t = self.q_sample(tgt.reshape(-1, tgt.shape[-1]), t, mask=mask.reshape(-1)).reshape(*tgt.shape)
        else:
            t, weights = self.schedule_sampler.sample(tgt.shape[0], tgt.device)
            x_t = self.q_sample(tgt, t, mask=mask)
        return x_t, t

    def forward(self, item_rep, item_tag, mask_seq, mask_tag):
        item_rep = self.encode_seq(item_rep, mask_seq)
        x_t, t = self.independent_diffuse(item_tag, mask_tag, self.independent_diffusion)
        if self.cfg_scale != 1:
            mask = torch.rand([mask_seq.shape[0], 1, 1], device=item_rep.device) > 0.7
            item_rep = torch.where(mask, torch.zeros_like(item_rep), item_rep)
        denoised_seq = self.net(item_rep, x_t, self._scale_timesteps(t), mask_seq, mask_tag)
        losses = F.mse_loss(denoised_seq, item_tag, reduction='none') * (mask_tag / mask_tag.sum(1, keepdim=True)).unsqueeze(-1)
        losses = losses.sum(1).mean()
        return denoised_seq, losses
