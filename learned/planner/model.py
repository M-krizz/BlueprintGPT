"""Planner transformer that predicts centroid priors and adjacency intent."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from learned.planner.data import PLANNER_PLOT_TYPES, PLANNER_ROOM_TYPES


@dataclass
class PlannerTransformerConfig:
    room_vocab_size: int = len(PLANNER_ROOM_TYPES) + 1
    plot_vocab_size: int = len(PLANNER_PLOT_TYPES) + 1
    max_rooms: int = 20
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    dropout: float = 0.1


class PlannerTransformer(nn.Module):
    def __init__(self, cfg: PlannerTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.room_emb = nn.Embedding(cfg.room_vocab_size, cfg.d_model, padding_idx=0)
        self.plot_emb = nn.Embedding(cfg.plot_vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_rooms, cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)

        self.centroid_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 2),
        )
        self.area_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )
        self.adj_q = nn.Linear(cfg.d_model, cfg.d_model)
        self.adj_k = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, room_type_ids: torch.Tensor, plot_type_ids: torch.Tensor, room_mask: torch.Tensor):
        batch_size, room_count = room_type_ids.shape
        positions = torch.arange(room_count, device=room_type_ids.device).unsqueeze(0).expand(batch_size, room_count)

        hidden = (
            self.room_emb(room_type_ids)
            + self.pos_emb(positions)
            + self.plot_emb(plot_type_ids).unsqueeze(1)
        )
        hidden = self.encoder(hidden, src_key_padding_mask=~room_mask.bool())
        hidden = self.norm(hidden)

        centroid = torch.sigmoid(self.centroid_head(hidden))
        area_ratio = torch.sigmoid(self.area_head(hidden)).squeeze(-1)

        q = self.adj_q(hidden)
        k = self.adj_k(hidden)
        adjacency_logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.cfg.d_model)

        invalid = (~room_mask.bool()).unsqueeze(1) | (~room_mask.bool()).unsqueeze(2)
        adjacency_logits = adjacency_logits.masked_fill(invalid, -10.0)

        diagonal = torch.eye(room_count, device=room_type_ids.device, dtype=torch.bool).unsqueeze(0)
        adjacency_logits = adjacency_logits.masked_fill(diagonal, -10.0)

        return {
            "centroid": centroid,
            "area_ratio": area_ratio,
            "adjacency_logits": adjacency_logits,
        }
