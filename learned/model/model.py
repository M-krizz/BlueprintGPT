"""
model.py – Decoder-only LayoutTransformer for autoregressive floor-plan generation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LayoutTransformerConfig:
    vocab_size: int = 293
    max_seq_len: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_model: int = 256
    d_ff: int = 1024
    dropout: float = 0.1
    pad_token: int = 0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: LayoutTransformerConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.proj_drop = nn.Dropout(cfg.dropout)
        # Upper-triangular mask: 1 = position to BLOCK (future tokens)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(cfg.max_seq_len, cfg.max_seq_len), diagonal=1),
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.mask[:T, :T].unsqueeze(0).unsqueeze(0).bool(), float("-inf"))
        att = self.attn_drop(att.softmax(dim=-1))
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: LayoutTransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class LayoutTransformer(nn.Module):
    def __init__(self, cfg: LayoutTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init_weights)
        n = sum(p.numel() for p in self.parameters())
        print(f"LayoutTransformer: {n:,} parameters")

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, f"Seq length {T} > max {self.cfg.max_seq_len}"
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.cfg.vocab_size),
                shift_targets.view(-1),
                ignore_index=self.cfg.pad_token,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=200, temperature=0.9,
                 top_p=0.95, top_k=0, eos_token=2,
                 logit_processor=None,
                 enforce_structure=True,
                 room_token=3,
                 type_token_start=4,
                 type_token_end=31,
                 coord_token_start=None,
                 coord_token_end=None,
                 min_rooms=1,
                 min_x_bin_gap=1,
                 min_y_bin_gap=1):
        """Autoregressive sampling with optional *logit_processor* callback.

        Parameters
        ----------
        logit_processor : callable(logits, seq) → logits, optional
            If given, called **before** top-k / top-p filtering with the raw
            logits tensor (shape [1, V]) and the current full sequence tensor.
        """

        if coord_token_start is None:
            coord_token_start = max(self.cfg.vocab_size - 256, 0)
        if coord_token_end is None:
            coord_token_end = self.cfg.vocab_size

        def _structured_mask(logits_now, seq_now):
            """Finite-state grammar mask:
            ROOM|EOS -> TYPE -> COORD -> COORD -> COORD -> COORD -> ROOM|EOS.
            """
            tokens = seq_now.squeeze(0).tolist()

            # Find where room chunks begin; prompt prefix tokens are ignored.
            last_room_idx = -1
            for idx in range(len(tokens) - 1, -1, -1):
                if tokens[idx] == room_token:
                    last_room_idx = idx
                    break

            if last_room_idx == -1:
                phase = "room_or_eos"
            else:
                tail_len = len(tokens) - last_room_idx - 1
                if tail_len == 0:
                    phase = "type"
                elif 1 <= tail_len <= 4:
                    phase = "coord"
                else:
                    phase = "room_or_eos"

            allowed = torch.zeros_like(logits_now, dtype=torch.bool)
            if phase == "room_or_eos":
                room_count = sum(1 for t in tokens if t == room_token)
                if room_count >= min_rooms:
                    allowed[:, eos_token] = True
                if 0 <= room_token < logits_now.size(-1):
                    allowed[:, room_token] = True
            elif phase == "type":
                ts = max(0, type_token_start)
                te = min(type_token_end, logits_now.size(-1))
                if te > ts:
                    allowed[:, ts:te] = True
            else:  # coord
                cs = max(0, coord_token_start)
                ce = min(coord_token_end, logits_now.size(-1))

                # Monotonic box constraints when sampling x2 / y2:
                # tail_len == 1  -> about to sample x1
                # tail_len == 2  -> about to sample y1
                # tail_len == 3  -> about to sample x2 (after ROOM,TYPE,x1,y1)
                # tail_len == 4  -> about to sample y2
                if last_room_idx >= 0:
                    tail_len = len(tokens) - last_room_idx - 1
                    if tail_len == 1:
                        ce = min(ce, coord_token_end - max(0, min_x_bin_gap))
                    elif tail_len == 2:
                        ce = min(ce, coord_token_end - max(0, min_y_bin_gap))
                    elif tail_len == 3 and last_room_idx + 2 < len(tokens):
                        x1_tok = tokens[last_room_idx + 2]
                        cs = max(cs, x1_tok + max(0, min_x_bin_gap))
                    elif tail_len == 4 and last_room_idx + 3 < len(tokens):
                        y1_tok = tokens[last_room_idx + 3]
                        cs = max(cs, y1_tok + max(0, min_y_bin_gap))

                if ce > cs:
                    allowed[:, cs:ce] = True

            logits_now = logits_now.masked_fill(~allowed, float("-inf"))

            # Safety valve: never leave an all -inf row.
            valid_any = torch.isfinite(logits_now).any(dim=-1)
            if not bool(valid_any.item()):
                logits_now.fill_(float("-inf"))
                logits_now[:, eos_token] = 0.0
            return logits_now

        self.eval()
        seq = prompt.clone()
        for _ in range(max_new_tokens):
            inp = seq[:, -self.cfg.max_seq_len:]
            logits, _ = self(inp)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if enforce_structure:
                logits = _structured_mask(logits, seq)

            # ── External constraint mask ──────────────────────────────────
            if logit_processor is not None:
                logits = logit_processor(logits, seq)

            # Top-k filtering
            if top_k > 0:
                v, _ = logits.topk(min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = logits.sort(descending=True)
                cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                # Shift right so first token is never removed
                remove = torch.cat([
                    torch.zeros_like(cum_probs[:, :1]),
                    cum_probs[:, :-1]
                ], dim=1) >= top_p
                sorted_logits[remove] = float("-inf")
                # Scatter back to original order
                logits = torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)

            probs = logits.softmax(dim=-1)
            # Guard against degenerate distributions
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                probs = torch.ones_like(probs) / probs.size(-1)
            nxt = torch.multinomial(probs, 1)
            seq = torch.cat([seq, nxt], dim=1)
            if nxt.item() == eos_token:
                break
        return seq
