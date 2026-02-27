"""
Обучение EN->RU переводчика на базе deepvk/RuModernBERT-small.

Энкодер: полный deepvk/RuModernBERT-small
Декодер: свои слои в стиле ModernBERT, веса инициализированы из encoder

Датасет: Helsinki-NLP/opus-100, конфиг en-ru
"""

import os
import sys
import glob
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForSeq2Seq,
    set_seed,
)
from transformers.modeling_outputs import Seq2SeqLMOutput
from torch.optim.lr_scheduler import LambdaLR
import sacrebleu

# ======Settings=========
MODEL_NAME = "deepvk/RuModernBERT-small"
DECODER_NUM_LAYERS = 6

DATASET_NAME = "Helsinki-NLP/tatoeba_mt_train"
DATASET_CONFIG = "eng-rus"
DATASET_SPLIT = "train"
SRC_FIELD = "source_text"
TGT_FIELD = "target_text"
MAX_CHAR_LENGTH = 2000  # Фильтруем строки длиннее

MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 512

OUTPUT_DIR = "models"
SAVE_STEPS = 100000
LOGGING_STEPS = 50

SEED = 42
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.01
DECAY_RATIO = 0.2
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 12
GRADIENT_ACCUMULATION_STEPS = 1

# Token IDs
PAD_TOKEN_ID = 50283      # [PAD]
EOS_TOKEN_ID = 1          # <|endoftext|>
DECODER_START_TOKEN_ID = 0  # <|padding|>
# ======Settings=========


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings."""
    # x: (batch, heads, seq, head_dim)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def get_rope_embeddings(seq_len: int, head_dim: int, device: torch.device, base: float = 160000.0):
    """Compute RoPE sin/cos embeddings."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)  # (seq, head_dim/2)
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim/2)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)
    return cos, sin


class CausalSelfAttention(nn.Module):
    """Causal self-attention в стиле ModernBERT."""
    
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Объединённый QKV как в ModernBERT
        self.Wqkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # QKV projection
        qkv = self.Wqkv(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # RoPE
        cos, sin = get_rope_embeddings(seq_len, self.head_dim, hidden_states.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.Wo(attn_output)


class CrossAttention(nn.Module):
    """Cross-attention к encoder."""
    
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Раздельные проекции для cross-attention
        self.Wq = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wkv = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        encoder_seq_len = encoder_hidden_states.shape[1]
        
        # Q from decoder, K/V from encoder
        q = self.Wq(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.Wkv(encoder_hidden_states).view(batch_size, encoder_seq_len, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        # Attention (без RoPE для cross-attention)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if encoder_attention_mask is not None:
            # (batch, enc_seq) -> (batch, 1, 1, enc_seq)
            attn_mask = encoder_attention_mask[:, None, None, :]
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float("-inf"))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.Wo(attn_output)


class GeGLU(nn.Module):
    """GeGLU MLP как в ModernBERT (GELU activation)."""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.Wi = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.Wo = nn.Linear(intermediate_size, hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.Wi(x)
        gate, up = x.chunk(2, dim=-1)
        return self.Wo(F.gelu(gate) * up)


class DecoderLayer(nn.Module):
    """Decoder layer: self-attn + cross-attn + FFN (pre-norm)."""
    
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        
        # Self-attention
        self.attn_norm = nn.LayerNorm(hidden_size, eps=1e-5, bias=False)
        self.self_attn = CausalSelfAttention(hidden_size, num_heads)
        
        # Cross-attention
        self.cross_attn_norm = nn.LayerNorm(hidden_size, eps=1e-5, bias=False)
        self.cross_attn = CrossAttention(hidden_size, num_heads)
        
        # FFN
        self.mlp_norm = nn.LayerNorm(hidden_size, eps=1e-5, bias=False)
        self.mlp = GeGLU(hidden_size, intermediate_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention (pre-norm)
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        # Cross-attention (pre-norm)
        residual = hidden_states
        hidden_states = self.cross_attn_norm(hidden_states)
        hidden_states = self.cross_attn(hidden_states, encoder_hidden_states, encoder_attention_mask)
        hidden_states = residual + hidden_states
        
        # FFN (pre-norm)
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class EncoderDecoderConfig(PretrainedConfig):
    model_type = "modernbert_enc_dec"

    def __init__(
        self,
        vocab_size: int = 50368,
        hidden_size: int = 384,
        num_attention_heads: int = 6,
        intermediate_size: int = 576,
        decoder_num_layers: int = 6,
        pad_token_id: int = PAD_TOKEN_ID,
        decoder_start_token_id: int = DECODER_START_TOKEN_ID,
        eos_token_id: int = EOS_TOKEN_ID,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.decoder_num_layers = decoder_num_layers
        self.num_hidden_layers = decoder_num_layers
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.eos_token_id = eos_token_id


class EncoderDecoderModel(PreTrainedModel):
    config_class = EncoderDecoderConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: EncoderDecoderConfig):
        super().__init__(config)
        self.config = config
        
        # Encoder (будет установлен позже)
        self.encoder = None
        
        # Decoder
        self.decoder_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.decoder_embed_norm = nn.LayerNorm(config.hidden_size, eps=1e-5, bias=False)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config.hidden_size, config.num_attention_heads, config.intermediate_size)
            for _ in range(config.decoder_num_layers)
        ])
        self.decoder_final_norm = nn.LayerNorm(config.hidden_size, eps=1e-5, bias=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def set_encoder(self, encoder: nn.Module):
        self.encoder = encoder

    def tie_weights(self):
        self.lm_head.weight = self.decoder_embed.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        # Encoder
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_hidden_states = encoder_outputs.last_hidden_state

        # Decoder input
        if decoder_input_ids is None:
            if labels is not None:
                decoder_input_ids = self._shift_right(labels)
            else:
                raise ValueError("Either decoder_input_ids or labels must be provided")

        # Decoder embeddings
        hidden_states = self.decoder_embed(decoder_input_ids)
        hidden_states = self.decoder_embed_norm(hidden_states)
        batch_size, seq_len = decoder_input_ids.shape
        
        # Causal mask
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=hidden_states.device),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        # Decoder layers
        for layer in self.decoder_layers:
            hidden_states = layer(
                hidden_states,
                encoder_hidden_states,
                attention_mask=causal_mask,
                encoder_attention_mask=attention_mask,
            )

        hidden_states = self.decoder_final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Label smoothing (0.1) реализован здесь, не в Trainer
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size), 
                labels.view(-1), 
                ignore_index=-100,
                label_smoothing=0.1
            )

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            encoder_last_hidden_state=encoder_hidden_states,
        )

    def _shift_right(self, input_ids: torch.Tensor) -> torch.Tensor:
        shifted = input_ids.new_zeros(input_ids.shape)
        shifted[:, 1:] = input_ids[:, :-1].clone()
        shifted[:, 0] = self.config.decoder_start_token_id
        shifted = shifted.masked_fill(shifted == -100, self.config.pad_token_id)
        return shifted


def init_decoder_from_encoder(model: EncoderDecoderModel, encoder_model: nn.Module):
    """Инициализирует веса decoder из encoder."""
    # Берём чередующиеся слои: 1,3,5,7,9,11 (через один)
    all_layers = list(encoder_model.layers)
    encoder_layers = [all_layers[i] for i in [1, 3, 5, 7, 9, 11]]
    
    for dec_layer, enc_layer in zip(model.decoder_layers, encoder_layers):
        # Self-attention: копируем Wqkv и Wo
        dec_layer.self_attn.Wqkv.weight.data.copy_(enc_layer.attn.Wqkv.weight.data)
        dec_layer.self_attn.Wo.weight.data.copy_(enc_layer.attn.Wo.weight.data)
        dec_layer.attn_norm.weight.data.copy_(enc_layer.attn_norm.weight.data)
        
        # Cross-attention: оставляем случайную инициализацию (не копируем)
        # cross_attn_norm тоже случайно
        
        # MLP
        dec_layer.mlp.Wi.weight.data.copy_(enc_layer.mlp.Wi.weight.data)
        dec_layer.mlp.Wo.weight.data.copy_(enc_layer.mlp.Wo.weight.data)
        dec_layer.mlp_norm.weight.data.copy_(enc_layer.mlp_norm.weight.data)
    
    # Embeddings (копируем только существующие токены, новые остаются случайными)
    src_embed = encoder_model.embeddings.tok_embeddings.weight.data
    model.decoder_embed.weight.data[:src_embed.shape[0]].copy_(src_embed)
    model.decoder_embed_norm.weight.data.copy_(encoder_model.embeddings.norm.weight.data)
    
    # Final norm
    model.decoder_final_norm.weight.data.copy_(encoder_model.final_norm.weight.data)


def build_model(tokenizer, resize_embeddings: bool = False) -> EncoderDecoderModel:
    """Собираем модель."""
    encoder = AutoModel.from_pretrained(MODEL_NAME)
    encoder_config = AutoConfig.from_pretrained(MODEL_NAME)
    
    # Ещё одна копия для инициализации decoder
    encoder_for_init = AutoModel.from_pretrained(MODEL_NAME)
    
    # Resize encoder embeddings если добавили токены
    if resize_embeddings:
        old_vocab_size = encoder_config.vocab_size
        new_vocab_size = len(tokenizer)
        print(f"Расширяем embeddings: {old_vocab_size} → {new_vocab_size}")
        encoder.resize_token_embeddings(new_vocab_size)

    # Используем токены из tokenizer
    config = EncoderDecoderConfig(
        vocab_size=len(tokenizer),
        hidden_size=encoder_config.hidden_size,
        num_attention_heads=encoder_config.num_attention_heads,
        intermediate_size=encoder_config.intermediate_size,
        decoder_num_layers=DECODER_NUM_LAYERS,
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model = EncoderDecoderModel(config)
    model.set_encoder(encoder)
    
    # Инициализируем decoder из encoder
    init_decoder_from_encoder(model, encoder_for_init)
    
    # Если resize, копируем ВСЕ embeddings из resized encoder (включая новые токены)
    if resize_embeddings:
        model.decoder_embed.weight.data.copy_(encoder.embeddings.tok_embeddings.weight.data)
        print(f"  decoder_embed скопирован из resized encoder")
    
    # Tie weights
    model.tie_weights()

    return model


def _preprocess_batch(examples: Dict, tokenizer) -> Dict[str, List[List[int]]]:
    src_texts = examples[SRC_FIELD]
    tgt_texts = examples[TGT_FIELD]

    # Source: стандартная токенизация
    model_inputs = tokenizer(
        src_texts,
        max_length=MAX_SOURCE_LENGTH,
        truncation=True,
    )

    # Target: без special tokens + EOS в конце
    labels_raw = tokenizer(
        tgt_texts,
        max_length=MAX_TARGET_LENGTH - 1,
        truncation=True,
        add_special_tokens=False,
    )
    eos_id = tokenizer.eos_token_id
    model_inputs["labels"] = [ids + [eos_id] for ids in labels_raw["input_ids"]]
    return model_inputs


def get_trapezoid_schedule(optimizer, num_warmup_steps: int, num_training_steps: int, num_decay_steps: int):
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        decay_start = num_training_steps - num_decay_steps
        if step >= decay_start:
            return max(0.0, (num_training_steps - step) / max(1, num_decay_steps))
        return 1.0
    return LambdaLR(optimizer, lr_lambda)


class TrapezoidTrainer(Trainer):
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if optimizer is None:
            optimizer = self.optimizer
        num_warmup = int(num_training_steps * WARMUP_RATIO)
        num_decay = int(num_training_steps * DECAY_RATIO)
        self.lr_scheduler = get_trapezoid_schedule(optimizer, num_warmup, num_training_steps, num_decay)
        return self.lr_scheduler


def greedy_decode_for_eval(model, input_ids, attention_mask, max_length, eos_token_id, decoder_start_token_id):
    """Greedy decoding для валидации."""
    batch_size = input_ids.shape[0]
    device = input_ids.device
    
    decoder_input_ids = torch.full((batch_size, 1), decoder_start_token_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_length - 1):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        
        next_token_logits = outputs.logits[:, -1, :]
        next_tokens = next_token_logits.argmax(dim=-1)
        next_tokens = next_tokens.masked_fill(finished, decoder_start_token_id)
        decoder_input_ids = torch.cat([decoder_input_ids, next_tokens.unsqueeze(-1)], dim=-1)
        
        finished = finished | (next_tokens == eos_token_id)
        if finished.all():
            break

    return decoder_input_ids


class WMT13ValidationCallback(TrainerCallback):
    """Callback для валидации на WMT13 при сохранении."""
    
    def __init__(self, tokenizer, device="cuda", max_samples=500, log_dir="models"):
        self.tokenizer = tokenizer
        self.device = device
        self.max_samples = max_samples
        self.sources = None
        self.references = None
        self.writer = None
        self.log_dir = log_dir
        self._load_wmt13()
    
    def _load_wmt13(self):
        """Загружаем WMT13 testset."""
        try:
            for langpair in ["en-ru", "ru-en"]:
                try:
                    src_file = sacrebleu.get_source_file("wmt13", langpair)
                    ref_file = sacrebleu.get_reference_files("wmt13", langpair)[0]
                    
                    with open(src_file, "r", encoding="utf-8") as f:
                        src_lines = [line.strip() for line in f]
                    with open(ref_file, "r", encoding="utf-8") as f:
                        ref_lines = [line.strip() for line in f]
                    
                    if langpair == "ru-en":
                        src_lines, ref_lines = ref_lines, src_lines
                    
                    # Ограничиваем количество примеров для скорости
                    self.sources = src_lines[:self.max_samples]
                    self.references = ref_lines[:self.max_samples]
                    print(f"WMT13 загружен: {len(self.sources)} примеров для валидации")
                    return
                except:
                    continue
            print("Не удалось загрузить WMT13")
        except Exception as e:
            print(f"Ошибка загрузки WMT13: {e}")
    
    def on_save(self, args, state, control, model=None, **kwargs):
        """Запускаем валидацию при сохранении."""
        if self.sources is None:
            return
        
        print(f"\n{'='*50}")
        print(f"Валидация на WMT13 (step {state.global_step})...")
        
        model.eval()
        hypotheses = []
        
        eos_id = model.config.eos_token_id
        bos_id = model.config.decoder_start_token_id
        
        with torch.no_grad():
            for i in range(0, len(self.sources), 8):  # batch=8
                batch_texts = self.sources[i:i+8]
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                ).to(self.device)
                
                output_ids = greedy_decode_for_eval(
                    model,
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    max_length=256,
                    eos_token_id=eos_id,
                    decoder_start_token_id=bos_id,
                )
                
                translations = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                hypotheses.extend(translations)
        
        # Считаем BLEU
        bleu = sacrebleu.corpus_bleu(hypotheses, [self.references[:len(hypotheses)]])
        chrf = sacrebleu.corpus_chrf(hypotheses, [self.references[:len(hypotheses)]])
        
        print(f"WMT13 BLEU: {bleu.score:.1f} | chrF: {chrf.score:.1f}")
        
        # Логируем в TensorBoard
        if self.writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, "runs"))
        
        self.writer.add_scalar("eval/wmt13_bleu", bleu.score, state.global_step)
        self.writer.add_scalar("eval/wmt13_chrf", chrf.score, state.global_step)
        self.writer.flush()
        
        # Примеры
        print("--- Примеры ---")
        for j in range(min(3, len(hypotheses))):
            print(f"SRC: {self.sources[j][:60]}...")
            print(f"REF: {self.references[j][:60]}...")
            print(f"HYP: {hypotheses[j][:60] if hypotheses[j] else '[ПУСТО]'}...")
            print()
        print('='*50 + "\n")
        
        model.train()


def load_model_from_checkpoint(checkpoint_path: str):
    """Загружает модель из чекпоинта для resume."""
    import safetensors.torch
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    config = EncoderDecoderConfig.from_pretrained(checkpoint_path)
    
    model = EncoderDecoderModel(config)
    
    # Загружаем encoder
    encoder = AutoModel.from_pretrained(MODEL_NAME)
    if encoder.embeddings.tok_embeddings.weight.shape[0] != config.vocab_size:
        encoder.resize_token_embeddings(config.vocab_size)
    model.set_encoder(encoder)
    
    # Загружаем веса модели
    weights_path = os.path.join(checkpoint_path, "model.safetensors")
    if os.path.exists(weights_path):
        state_dict = safetensors.torch.load_file(weights_path)
    else:
        weights_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location="cpu")
    
    model.load_state_dict(state_dict, strict=False)
    model.tie_weights()
    
    return model, tokenizer


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed(SEED)

    # Проверяем --resume сразу
    resume_from = None
    if "--resume" in sys.argv:
        checkpoints = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
            resume_from = checkpoints[-1]
            print(f"Resume mode: загружаем из {resume_from}")
    
    if resume_from:
        # Загружаем модель из чекпоинта
        model, tokenizer = load_model_from_checkpoint(resume_from)
        print(f"Модель загружена из чекпоинта")
    else:
        # Создаём модель с нуля
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        
        # Настраиваем special tokens в стиле T5
        tokenizer.pad_token = "[PAD]"
        tokenizer.pad_token_id = PAD_TOKEN_ID
        tokenizer.eos_token = "</s>"
        tokenizer.bos_token = "<s>"
        
        # Добавляем новые токены если их нет
        special_tokens = {"eos_token": "</s>", "bos_token": "<s>"}
        num_added = tokenizer.add_special_tokens(special_tokens)
        print(f"Добавлено special tokens: {num_added}")
        print(f"  pad_token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
        print(f"  eos_token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
        print(f"  bos_token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")

        model = build_model(tokenizer, resize_embeddings=(num_added > 0))

    # Загружаем датасет (streaming + ограничение 100M)
    MAX_SAMPLES = 100_000_000
    print(f"Загрузка датасета {DATASET_NAME} ({DATASET_CONFIG}), streaming mode...")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT, streaming=True)
    
    # Фильтруем по длине, shuffle и берём первые 100M
    ds = ds.filter(lambda x: len(x[SRC_FIELD]) <= MAX_CHAR_LENGTH and len(x[TGT_FIELD]) <= MAX_CHAR_LENGTH)
    ds = ds.shuffle(seed=SEED, buffer_size=10000)
    ds = ds.take(MAX_SAMPLES)
    print(f"Используем до {MAX_SAMPLES:,} строк (streaming, shuffled)")

    # Collator с токенизацией на лету
    class TokenizingCollator:
        def __init__(self, tokenizer, max_src_len, max_tgt_len):
            self.tokenizer = tokenizer
            self.max_src_len = max_src_len
            self.max_tgt_len = max_tgt_len
        
        def __call__(self, features):
            src_texts = [f[SRC_FIELD] for f in features]
            tgt_texts = [f[TGT_FIELD] for f in features]
            
            # Токенизируем source
            model_inputs = self.tokenizer(
                src_texts,
                max_length=self.max_src_len,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            
            # Токенизируем target БЕЗ padding (чтобы добавить EOS правильно)
            labels_raw = self.tokenizer(
                tgt_texts,
                max_length=self.max_tgt_len - 1,
                truncation=True,
                add_special_tokens=False,
            )
            
            # Добавляем EOS в конец каждой последовательности
            eos_id = self.tokenizer.eos_token_id
            labels_with_eos = [ids + [eos_id] for ids in labels_raw["input_ids"]]
            
            # Паддим до максимальной длины в batch
            max_len = max(len(ids) for ids in labels_with_eos)
            labels_padded = []
            for ids in labels_with_eos:
                # Паддим -100 (игнорируется в loss)
                padded = ids + [-100] * (max_len - len(ids))
                labels_padded.append(padded)
            
            model_inputs["labels"] = torch.tensor(labels_padded, dtype=torch.long)
            return model_inputs
    
    data_collator = TokenizingCollator(tokenizer, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH)

    # Для streaming dataset нужен max_steps
    effective_batch = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    max_steps = MAX_SAMPLES // effective_batch
    print(f"Max steps: {max_steps:,} (batch={effective_batch})")
    
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        do_train=True,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        max_steps=max_steps,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        # label_smoothing реализован в модели, не здесь (иначе Trainer удаляет labels)
        fp16=torch.cuda.is_available(),
        report_to=["tensorboard"],
        remove_unused_columns=False,  # Нужно для streaming
        dataloader_pin_memory=False,  # Отключаем для streaming
        dataloader_num_workers=16,
    )

    # Callback для валидации на WMT13
    wmt13_callback = WMT13ValidationCallback(tokenizer, device="cuda" if torch.cuda.is_available() else "cpu")
    
    trainer = TrapezoidTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[wmt13_callback],
    )

    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
