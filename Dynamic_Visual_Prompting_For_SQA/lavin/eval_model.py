# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F
import clip
import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)
from lavin.model import AdapterMLP

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    hidden_proj: int=128

    max_batch_size: int = 32
    max_seq_len: int = 2048

    insert_layer: int = 4



class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).cuda()
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).cuda()
        self.gate = torch.nn.Parameter(torch.zeros(1, self.n_local_heads, 1, 1))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        #add modilaty embedding
        if start_pos==0:
            self.cache_k[:bsz, start_pos : start_pos + seqlen-1] = xk[:,1:]
            self.cache_v[:bsz, start_pos : start_pos + seqlen-1] = xv[:,1:]

            keys = xk
            values = xv
        else:
            self.cache_k[:bsz, start_pos: start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos: start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]


        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)

        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.drop_path =  nn.Identity()
        self.cache_weights = torch.zeros(
            (args.max_batch_size, 2)
        ).cuda()
        self.cache_weights_ffn = torch.zeros(
            (args.max_batch_size, 2)
        ).cuda()
    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, adapter)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Cross_Attention(nn.Module):
    def __init__(self, clip_size, llama_size, mid_size, drop_rate, num_heads):
        super(Cross_Attention, self).__init__()
        self.num_heads = num_heads
        self.clip_size = clip_size
        self.llama_size = llama_size
        self.mid_size = mid_size

        self.linear_v = nn.Linear(clip_size, mid_size)
        self.linear_k = nn.Linear(clip_size, mid_size)
        self.linear_q = nn.Linear(llama_size, mid_size)
        self.linear_merge = nn.Linear(mid_size, llama_size)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.num_heads,
            int(self.mid_size / self.num_heads)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.num_heads,
            int(self.mid_size / self.num_heads)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.num_heads,
            int(self.mid_size / self.num_heads)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.mid_size
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

from  torch.cuda.amp import autocast
class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

        self.backbone = clip.load('ViT-L/14')[0]

        self.adapter_proj = AdapterMLP(1024, params.hidden_proj, params.dim).float()
        self.adapter_modality_embedding=nn.Embedding(2,params.dim).float()

        self.insert_layer = params.insert_layer

        self.image_params = nn.Parameter(torch.randn(1, params.dim))

        self.cross_attn = Cross_Attention(clip_size=1024,
                                          llama_size=params.dim,
                                          mid_size=512,
                                          drop_rate=0.1,
                                          num_heads=16).float()
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def generate_query_prompt(self, h, image_prompt_index, text_prompt_index, prompt_len, img_indicators):
        process_prompts = []
        for i, (lenth, indicator) in enumerate(zip(prompt_len, img_indicators)):
            current_h = h[i]
            if indicator > 0:
                process_prompt = current_h[image_prompt_index:image_prompt_index + lenth]
                process_prompt = self.pooling(process_prompt.permute(1, 0)).permute(1, 0)
                process_prompts.append(process_prompt.unsqueeze(0))
            else:
                process_prompt = current_h[text_prompt_index:text_prompt_index + lenth]
                process_prompt = self.pooling(process_prompt.permute(1, 0)).permute(1, 0)
                process_prompts.append(process_prompt.unsqueeze(0))
        process_prompts = torch.cat(process_prompts, 0)
        return process_prompts

    def replace_dvp(self, h, dvp, image_prompt_index, img_indicators):
        _bsz, seqlen, _ = h.shape
        new_examples = []
        for i, example in enumerate(h):
            if img_indicators[i] > 0.:
                new_example = torch.cat([example[:image_prompt_index-1], dvp[i], example[image_prompt_index:]], 0)
            else:
                new_example = example
            new_examples.append(new_example.unsqueeze(0))
        new_examples = torch.cat(new_examples, 0)
        return new_examples

    def create_front_mask(self,mask,_bsz,image_prompt_index,img_indicators):
        front_mask = mask
        for i in range(_bsz):
            if img_indicators[i] > 0:
                front_mask[i,:,image_prompt_index:,image_prompt_index-1] = float("-inf")
            else:
                pass
        return front_mask

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int, indicators:torch.Tensor,image_prompt_index:int, text_prompt_index:int, prompt_len, image_feats):
        with autocast():
            _bsz, seqlen,_ = tokens.shape
            # h = self.tok_embeddings(tokens)
            h=tokens
            self.freqs_cis = self.freqs_cis.to(h.device)
            freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

            if seqlen > 1:
                mask = torch.full((_bsz, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
                mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
                # mask decision token
                mask[:, :, 1:, 0] = float("-inf")
                front_mask = self.create_front_mask(mask, _bsz, image_prompt_index, indicators)
                for layer in self.layers[:self.insert_layer]:
                    h = layer(h, start_pos, freqs_cis, front_mask)
                query_prompts = self.generate_query_prompt(h,
                                                           image_prompt_index,
                                                           text_prompt_index,
                                                           prompt_len,
                                                           indicators)
                dvp = self.cross_attn(v=image_feats,
                                      k=image_feats,
                                      q=query_prompts,
                                      mask=None)
                # after_insert_h = h.clone()
                after_insert_h = self.replace_dvp(h, dvp, image_prompt_index, indicators)
                for layer in self.layers[self.insert_layer:]:
                    after_insert_h = layer(after_insert_h, start_pos, freqs_cis, mask)

            else:
                front_mask = torch.zeros((_bsz, 1, seqlen, start_pos + seqlen), device=tokens.device)
                front_mask[:, :, :, image_prompt_index-2] = float("-inf")
                for layer in self.layers[:self.insert_layer]:
                    h = layer(h, start_pos, freqs_cis, front_mask)
                mask = None
                after_insert_h = h
                for layer in self.layers[self.insert_layer:]:
                    after_insert_h = layer(after_insert_h, start_pos, freqs_cis, mask)

            h = self.norm(after_insert_h)
            output = self.output(h[:, -1, :])  # only compute last logits
            return output.float()
