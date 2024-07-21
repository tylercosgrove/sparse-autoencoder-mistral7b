import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""

This implementation is largely based on 'Scaling and evaluating sparse autoencoders' (Gao et al. 2024)
arXiv link: https://arxiv.org/abs/2406.04093v1

"""


class SparseAutoencoder(nn.Module):
    def __init__(
        self,
        d_model,
        d_hidden,
        k=128,
        auxk=256,
        dead_steps_threshold=10_000_000,
        data_geometric_median=None,
    ):
        super().__init__()

        self.w_enc = nn.Parameter(torch.empty(d_model, d_hidden))
        self.w_dec = nn.Parameter(torch.empty(d_hidden, d_model))

        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.b_pre = nn.Parameter(torch.zeros(d_model))

        self.d_model = d_model
        self.d_hidden = d_hidden
        self.k = k
        self.auxk = auxk

        batch_size = 256
        self.dead_steps_threshold = dead_steps_threshold / batch_size

        # initialize weights
        if data_geometric_median is not None:
            self.b_pre.data = torch.load(data_geometric_median)

        # tied init
        nn.init.kaiming_uniform_(self.w_enc, a=math.sqrt(5))
        self.w_dec.data = self.w_enc.data.T.clone()
        self.w_dec.data /= self.w_dec.data.norm(dim=0)

        # Initialize dead neuron tracking
        self.register_buffer(
            "stats_last_nonzero", torch.zeros(d_hidden, dtype=torch.long)
        )

    def topK_activation(self, x, k):
        topk = torch.topk(x, k=k, dim=-1, sorted=False)
        values = F.relu(topk.values)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result

    def LN(self, x, eps=1e-5):
        mu = x.mean(dim=-1, keepdim=True)
        x = x - mu
        std = x.std(dim=-1, keepdim=True)
        x = x / (std + eps)
        return x, mu, std

    def auxk_mask_fn(self):
        dead_mask = self.stats_last_nonzero > self.dead_steps_threshold
        return dead_mask

    def forward(self, x):
        x, mu, std = self.LN(x)
        x = x - self.b_pre

        pre_acts = x @ self.w_enc + self.b_enc
        latents = self.topK_activation(pre_acts, k=self.k)

        self.stats_last_nonzero *= (latents == 0).all(dim=0).long()
        self.stats_last_nonzero += 1

        dead_mask = self.auxk_mask_fn()
        num_dead = dead_mask.sum().item()

        recons = latents @ self.w_dec + self.b_pre
        recons = recons * std + mu

        if num_dead > 0:
            k_aux = min(x.shape[-1] // 2, num_dead)

            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            auxk_acts = self.topK_activation(auxk_latents, k=k_aux)

            auxk = auxk_acts @ self.w_dec + self.b_pre
            auxk = auxk * std + mu
        else:
            auxk = None

        return recons, auxk, num_dead

    @torch.no_grad()
    def forward_val(self, x, features=None):
        x, mu, std = self.LN(x)
        x = x - self.b_pre
        pre_acts = x @ self.w_enc + self.b_enc
        latents = self.topK_activation(pre_acts, self.k)

        for feat in features:
            latents[:, feat["feat_index"]] = feat["val"]

        recons = latents @ self.w_dec + self.b_pre
        recons = recons * std + mu
        return recons

    @torch.no_grad()
    def norm_weights(self):
        self.w_dec.data /= self.w_dec.data.norm(dim=0)

    @torch.no_grad()
    def norm_grad(self):
        dot_products = torch.sum(self.w_dec.data * self.w_dec.grad, dim=0)
        self.w_dec.grad.sub_(self.w_dec.data * dot_products.unsqueeze(0))

    @torch.no_grad()
    def get_acts(self, x):
        x, mu, std = self.LN(x)
        x = x - self.b_pre
        pre_acts = x @ self.w_enc + self.b_enc
        latents = self.topK_activation(pre_acts, self.k)
        return latents
