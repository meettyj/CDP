from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn

from .gin import GIN
from .gat import GAT
from .gcn import GCN
from .dot_gat import DotGAT
from .loss_func import sce_loss
from cdp.utils import create_norm, drop_edge

import copy
import numpy as np

import torch.nn.functional as F


def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "dotgat":
        mod = DotGAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "mlp":
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError

    return mod


def cosine_loss(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class EMA:
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new
        beta = 1 - (1 - self.beta) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        self.step += 1
        return old * beta + (1 - beta) * new


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class Normalize(torch.nn.Module):
    def __init__(self, dim=None, norm='batch'):
        super().__init__()
        if dim is None or norm == 'none':
            self.norm = lambda x: x
        if norm == 'batch':
            self.norm = torch.nn.BatchNorm1d(dim)
        elif norm == 'layer':
            self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            loss_fn: str = "sce",
            edge_recon_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
            device: str = 'cpu',
            max_epoch: int = 100,
            args=None,
    ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self.edge_recon_rate = edge_recon_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden

        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden

        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        # noised encoder
        self.encoder_noised = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        # build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

        # embed_transformed
        self.layer_embed_transformed_origin = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
        self.layer_embed_transformed_ema = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        self.device = device

        # edge recon
        self.edge_recon_weight = args.edge_recon_weight
        self.encoder_to_decoder_edge_recon = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
        self.decoder_out_transformed_edge_recon = nn.Linear(in_dim, in_dim, bias=False)

        # contrastive
        self.contrastive_loss_weight = args.contrastive_loss_weight
        self.encoder_mapping_attr_pred = nn.Linear(dec_in_dim, dec_in_dim)
        self.encoder_mapping_contrastive_shared = nn.Sequential(nn.Linear(dec_in_dim, dec_in_dim), nn.ReLU(inplace=True), nn.Linear(dec_in_dim, dec_in_dim))
        self.encoder_bgrl = None

        # consistency
        self.encoder_to_decoder_bgrl = None
        self.decoder_bgrl = None
        self.consistency_loss_weight = args.consistency_loss_weight
        self.criterion_consistency_alpha_l = args.criterion_consistency_alpha_l
        self.criterion_consistency = self.setup_loss_fn("sce", self.criterion_consistency_alpha_l)

        # adaptive masking
        self.adaptive_masking_weight = args.adaptive_masking_weight
        self.num_attention_head = args.num_attention_head
        self.multi_head_attention_dropout_rate = args.multi_head_attention_dropout_rate
        self.attention_input_linear_query = nn.Sequential(nn.Linear(in_dim, dec_in_dim), nn.LeakyReLU(inplace=True), nn.Linear(dec_in_dim, dec_in_dim))
        self.attention_input_linear_key = nn.Sequential(nn.Linear(in_dim, dec_in_dim), nn.LeakyReLU(inplace=True), nn.Linear(dec_in_dim, dec_in_dim))
        self.attention_input_linear_value = nn.Sequential(nn.Linear(in_dim, dec_in_dim), nn.LeakyReLU(inplace=True), nn.Linear(dec_in_dim, dec_in_dim))
        self.multi_head_attention = nn.MultiheadAttention(dec_in_dim, self.num_attention_head, dropout=self.multi_head_attention_dropout_rate)
        self.attention_output_linear = nn.Sequential(nn.Linear(dec_in_dim, dec_in_dim), nn.LeakyReLU(inplace=True), nn.Linear(dec_in_dim, 1))
        self.attention_softmax = nn.Softmax(dim=-1)

        self.dataset = args.dataset

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def forward(self, g, x):
        loss_item = {}

        # ---- random masking, or adaptive masking ----
        if not self.adaptive_masking_weight > 0:
            use_g, use_x, mask_nodes, keep_nodes = self.get_masked_graph(g, x)
        else:
            assert self.adaptive_masking_weight > 0
            use_g, use_x, mask_nodes, keep_nodes, p_x = self.get_adaptive_masked_graph(g, x, self._mask_rate)

        # ---- attribute reconstruction ----
        loss_all, rep_attr_pred, origin_dec_out, _ = self.mask_attr_prediction(x, use_g, use_x, mask_nodes, keep_nodes)
        loss = loss_all.mean()

        # ---- update MultiheadAttention in adaptive masking ----
        if self.adaptive_masking_weight > 0:
            if self.dataset == 'ogbn-arxiv':
                iterate_gap = 2048
                iterate_num = int(p_x.shape[0] / iterate_gap) - 1
                probs_list = []
                for i in range(iterate_num):
                    cur_probs = p_x[i * iterate_gap: (i + 1) * iterate_gap] if i != iterate_num - 1 else p_x[i * iterate_gap:]

                    m = torch.distributions.categorical.Categorical(probs=cur_probs)

                    # log-probabilities
                    log_probs = m.log_prob(torch.arange(0, cur_probs.shape[0], 1).to(p_x.device))
                    probs_list.append(log_probs)

                all_probs = torch.cat(probs_list, dim=0)[mask_nodes].squeeze()
                loss_adaptive_masking = - torch.mean(all_probs * loss_all.detach())
                add_loss_adaptive_masking = self.adaptive_masking_weight * loss_adaptive_masking

                loss += add_loss_adaptive_masking
                loss_item["ada_m"] = add_loss_adaptive_masking.item()
            # other datasets
            else:
                # categorical distribution
                m = torch.distributions.categorical.Categorical(probs=p_x)

                # log-probabilities
                log_probs = m.log_prob(torch.arange(0, p_x.shape[0], 1).to(p_x.device))[mask_nodes]
                loss_adaptive_masking = - torch.mean(log_probs * loss_all.detach())
                add_loss_adaptive_masking = self.adaptive_masking_weight * loss_adaptive_masking

                loss += add_loss_adaptive_masking
                loss_item["ada_m"] = add_loss_adaptive_masking.item()

        # *** prepare edge drop for 1) edge reconstruction, or 2) contrastive
        if self.edge_recon_weight > 0 or self.contrastive_loss_weight > 0:
            assert self.edge_recon_rate > 0
            use_g_drop_edge, keep_edges_drop_edge, removed_edges_drop_edge = drop_edge(g, self.edge_recon_rate)

        # ---- edge reconstruction ----
        if self.edge_recon_weight > 0:
            loss_edge_recon, rep_edge_recon = self.mask_edge_reconstruction(use_g_drop_edge, removed_edges_drop_edge, x)
            add_loss_edge_recon = self.edge_recon_weight * loss_edge_recon
            loss += add_loss_edge_recon
            loss_item["edge"] = add_loss_edge_recon.item()

        # ---- contrastive ----
        if self.contrastive_loss_weight > 0:
            loss_contrastive_all, enc_ema_out = self.contrastive_ema_bgrl(use_g, use_g_drop_edge, x, rep_attr_pred, mask_nodes)
            loss_contrastive = loss_contrastive_all.mean()
            add_loss_contrastive = self.contrastive_loss_weight * loss_contrastive
            loss += add_loss_contrastive
            loss_item["con"] = add_loss_contrastive.item()

        # ---- consistency ----
        if self.consistency_loss_weight > 0:
            consistency_func_input = enc_ema_out if self.contrastive_loss_weight > 0 else (use_g, use_x)
            loss_consistency_all = self.cal_consistency_after_decoder(consistency_func_input, origin_dec_out, use_g, mask_nodes)
            loss_consistency = loss_consistency_all.mean()
            add_loss_consistency = self.consistency_loss_weight * loss_consistency
            loss_item["cst"] = add_loss_consistency.item()
            loss += add_loss_consistency

        loss_item["total"] = loss.item()

        return loss, loss_item

    def get_encoder_bgrl(self):
        if self.encoder_bgrl is None:
            self.encoder_bgrl = copy.deepcopy(self.encoder)

            for p in self.encoder_bgrl.parameters():
                p.requires_grad = False
        return self.encoder_bgrl

    def get_encoder_to_decoder_bgrl(self):
        if self.encoder_to_decoder_bgrl is None:
            self.encoder_to_decoder_bgrl = copy.deepcopy(self.encoder_to_decoder)

            for p in self.encoder_to_decoder_bgrl.parameters():
                p.requires_grad = False
        return self.encoder_to_decoder_bgrl

    def get_decoder_bgrl(self):
        if self.decoder_bgrl is None:
            self.decoder_bgrl = copy.deepcopy(self.decoder)

            for p in self.decoder_bgrl.parameters():
                p.requires_grad = False
        return self.decoder_bgrl

    def update_bgrl(self, momentum: float):
        # encoder
        for p, new_p in zip(self.get_encoder_bgrl().parameters(), self.encoder.parameters()):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

        # encoder_to_decoder
        for p, new_p in zip(self.get_encoder_to_decoder_bgrl().parameters(), self.encoder_to_decoder.parameters()):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

        # decoder
        for p, new_p in zip(self.get_decoder_bgrl().parameters(), self.decoder.parameters()):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

    def cal_consistency_after_decoder(self, consistency_func_input, origin_dec_out, pre_use_g, mask_nodes):

        if self.contrastive_loss_weight > 0:
            enc_out = consistency_func_input
        else:
            g, x = consistency_func_input
            cur_encoder_bgrl = self.get_encoder_bgrl()
            enc_out, _ = cur_encoder_bgrl(g, x, return_hidden=True)

        rep = self.get_encoder_to_decoder_bgrl()(enc_out)

        # re-mask
        if self._decoder_type not in ("mlp", "linear"):
            rep[mask_nodes] = 0

        cur_decoder_bgrl = self.get_decoder_bgrl()
        if self._decoder_type in ("mlp", "linear"):
            recon = cur_decoder_bgrl(rep)
        else:
            recon = cur_decoder_bgrl(pre_use_g, rep)

        recon = recon[mask_nodes]
        origin_dec_out = origin_dec_out[mask_nodes]

        loss_consistency = self.criterion_consistency(origin_dec_out, recon.detach())
        return loss_consistency

    def contrastive_ema_bgrl(self, masked_g, use_g_drop_edge, x, rep_attr_pred, mask_nodes):

        def origin_bgrl_loss(q1, q2, y1, y2):
            return 2 - F.cosine_similarity(q1, y2.detach(), dim=-1).mean() - F.cosine_similarity(q2, y1.detach(), dim=-1).mean()

        # aug
        h2, _ = self.encoder(use_g_drop_edge, x, return_hidden=True)

        # projection heads
        h1_pred = self.encoder_mapping_contrastive_shared(rep_attr_pred)
        h2_pred = self.encoder_mapping_contrastive_shared(h2)

        # ema
        with torch.no_grad():
            cur_encoder_bgrl = self.get_encoder_bgrl()
            h1_target, _ = cur_encoder_bgrl(masked_g, x, return_hidden=True)
            h2_target, _ = cur_encoder_bgrl(use_g_drop_edge, x, return_hidden=True)

        loss_contrastive = origin_bgrl_loss(h1_pred, h2_pred, h1_target.detach(), h2_target.detach())

        return loss_contrastive, h1_target.detach()

    # edge recon: generative
    def mask_edge_reconstruction(self, use_g_drop_edge, removed_edges, x):
        removed_edges_src, removed_edges_dst = removed_edges
        emb_drop_edge, _ = self.encoder(use_g_drop_edge, x, return_hidden=True)
        rep_drop_edge = self.encoder_to_decoder_edge_recon(emb_drop_edge)
        if self._decoder_type in ("mlp", "linear"):
            recon_feat = self.decoder(rep_drop_edge)
        else:
            recon_feat = self.decoder(use_g_drop_edge, rep_drop_edge)
        recon_feat_transformed = self.decoder_out_transformed_edge_recon(recon_feat)
        recon_feat_for_removed_edges_src = recon_feat_transformed[removed_edges_src]
        recon_feat_for_removed_edges_dst = recon_feat_transformed[removed_edges_dst]

        # pos and neg
        pos_score = torch.mul(recon_feat_for_removed_edges_src, recon_feat_for_removed_edges_dst).sum(dim=-1)
        shuffled_recon_feat_for_removed_edges_dst = recon_feat_for_removed_edges_dst[torch.randperm(recon_feat_for_removed_edges_dst.size()[0])]
        neg_score = torch.mul(recon_feat_for_removed_edges_src, shuffled_recon_feat_for_removed_edges_dst).sum(dim=-1)

        # margin loss
        loss_edge_recon = (neg_score - pos_score + 1).clamp(min=0).mean()

        return loss_edge_recon, emb_drop_edge

    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        perm = torch.randperm(num_nodes, device=x.device)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)

    def get_adaptive_masked_graph(self, g, x, mask_rate):
        num_nodes = g.num_nodes()
        visible_patches = int((1 - mask_rate) * num_nodes)

        query, key, value = x, x, x
        query = self.attention_input_linear_query(query)
        key = self.attention_input_linear_key(key)
        value = self.attention_input_linear_value(value)

        # 0 or 1
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)

        attn_output, attn_output_weights = self.multi_head_attention(query, key, value)
        attn_output = attn_output.squeeze()

        attn_output = self.attention_output_linear(attn_output)
        attn_output = attn_output.squeeze()

        p_x = self.attention_softmax(attn_output)

        vis_idx = torch.multinomial(p_x, num_samples=visible_patches, replacement=False)
        mask_nodes = torch.ones((x.shape[0])).to(x.device, non_blocking=True)
        mask_nodes.scatter_(dim=-1, index=vis_idx.long(), value=0.0)
        mask_nodes = mask_nodes.to(torch.bool)
        keep_nodes = ~mask_nodes

        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token

        return g, out_x, mask_nodes, keep_nodes, p_x

    def get_masked_graph(self, g, x):
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)
        return pre_use_g, use_x, mask_nodes, keep_nodes

    def mask_attr_prediction(self, x, pre_use_g, use_x, mask_nodes, keep_nodes):
        use_g = pre_use_g

        enc_out, all_hidden = self.encoder(use_g, use_x, return_hidden=True)
        if self._concat_hidden:
            enc_out = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_out)

        # re-mask
        if self._decoder_type not in ("mlp", "linear"):
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "linear"):
            recon = self.decoder(rep)
        else:
            recon = self.decoder(pre_use_g, rep)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init)

        return loss, enc_out, recon, mask_nodes

    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep

    def embed_transformed_origin(self, emb_origin):
        rep = self.layer_embed_transformed_origin(emb_origin)
        return rep

    def embed_transformed_ema(self, emb_ema):
        rep = self.layer_embed_transformed_ema(emb_ema)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
