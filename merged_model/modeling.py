from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput

try:
    from .configuration import EncoderDecoderConfig
except ImportError:
    from configuration import EncoderDecoderConfig


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def get_rope_embeddings(seq_len: int, head_dim: int, device: torch.device, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos().unsqueeze(0).unsqueeze(0), freqs.sin().unsqueeze(0).unsqueeze(0)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: EncoderDecoderConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rope_theta = config.rope_theta
        
        self.Wqkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=False)
        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        qkv = self.Wqkv(hidden_states).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        
        cos, sin = get_rope_embeddings(seq_len, self.head_dim, hidden_states.device, self.rope_theta)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.Wo(attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))


class CrossAttention(nn.Module):
    def __init__(self, config: EncoderDecoderConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        self.Wq = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.Wkv = nn.Linear(config.hidden_size, config.hidden_size * 2, bias=False)
        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        encoder_seq_len = encoder_hidden_states.shape[1]
        
        q = self.Wq(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.Wkv(encoder_hidden_states).view(batch_size, encoder_seq_len, 2, self.num_heads, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
        
        attn_mask = None
        if encoder_attention_mask is not None:
            attn_mask = encoder_attention_mask[:, None, None, :].expand(-1, self.num_heads, seq_len, -1).bool()
        
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.Wo(attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))


class GeGLU(nn.Module):
    def __init__(self, config: EncoderDecoderConfig):
        super().__init__()
        self.Wi = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.Wi(x).chunk(2, dim=-1)
        return self.Wo(F.gelu(gate) * up)


class DecoderLayer(nn.Module):
    def __init__(self, config: EncoderDecoderConfig):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=False)
        self.self_attn = CausalSelfAttention(config)
        self.cross_attn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=False)
        self.cross_attn = CrossAttention(config)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=False)
        self.mlp = GeGLU(config)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.attn_norm(hidden_states))
        hidden_states = hidden_states + self.cross_attn(self.cross_attn_norm(hidden_states), encoder_hidden_states, encoder_attention_mask)
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


class EncoderDecoderModel(PreTrainedModel, GenerationMixin):
    config_class = EncoderDecoderConfig
    main_input_name = "input_ids"
    _supports_cache_class = False
    _tied_weights_keys = ["lm_head.weight", "decoder_embed.weight"]

    def __init__(self, config: EncoderDecoderConfig):
        super().__init__(config)
        encoder_config = AutoConfig.for_model(**config.encoder_config)
        self.encoder = AutoModel.from_config(encoder_config)
        self.decoder_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.decoder_embed_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=False)
        self.decoder_layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.decoder_num_layers)])
        self.decoder_final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder_embed

    def set_input_embeddings(self, value):
        self.decoder_embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        encoder_hidden_states = encoder_outputs[0] if isinstance(encoder_outputs, (tuple, list)) else encoder_outputs.last_hidden_state

        if decoder_input_ids is None:
            if labels is not None:
                decoder_input_ids = self._shift_right(labels)
            else:
                raise ValueError("decoder_input_ids or labels required")

        hidden_states = self.decoder_embed_norm(self.decoder_embed(decoder_input_ids))

        for layer in self.decoder_layers:
            hidden_states = layer(hidden_states, encoder_hidden_states, attention_mask)

        logits = self.lm_head(self.decoder_final_norm(hidden_states))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1), ignore_index=-100)

        return Seq2SeqLMOutput(loss=loss, logits=logits, encoder_last_hidden_state=encoder_hidden_states)

    def _shift_right(self, input_ids: torch.Tensor) -> torch.Tensor:
        shifted = input_ids.new_zeros(input_ids.shape)
        shifted[:, 1:] = input_ids[:, :-1]
        shifted[:, 0] = self.config.decoder_start_token_id
        shifted.masked_fill_(shifted == -100, self.config.pad_token_id)
        return shifted

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_outputs: Optional[BaseModelOutput] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return {
            "decoder_input_ids": decoder_input_ids,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
        }

    def _prepare_encoder_decoder_kwargs_for_generation(self, inputs_tensor, model_kwargs, model_input_name, generation_config):
        encoder_kwargs = {k: v for k, v in model_kwargs.items() if not k.startswith("decoder_") and k not in ("labels",)}
        encoder_outputs = self.encoder(inputs_tensor, **encoder_kwargs)
        model_kwargs["encoder_outputs"] = encoder_outputs
        return model_kwargs
