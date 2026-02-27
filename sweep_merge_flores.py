"""
Скрипт: merge последних N чекпоинтов и оценка только на FLORES-200 devtest.
"""

import os
import shutil
from typing import List, Tuple

import torch
import safetensors.torch
import sacrebleu
from transformers import AutoTokenizer

import eval as eval_mod
from merge_checkpoints import get_checkpoints, load_state_dict, average_checkpoints

# ======Settings=========
MODELS_DIR = "models"
OUTPUT_ROOT = "merged_sweep"
RANGE_MAX = 12
RANGE_MIN = 2

# Параметры оценки
EVAL_BATCH_SIZE = 32
NUM_BEAMS = 4
TEMPERATURE = 0.0
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 512

# FLORES-200 devtest EN→RU
FLORES200_URL = "https://tinyurl.com/flores200dataset"
FLORES200_CACHE_DIR = "data/flores200"
FLORES200_SPLIT = "devtest"
FLORES200_SRC_LANG = "eng_Latn"
FLORES200_TGT_LANG = "rus_Cyrl"
# ======Settings=========


def merge_last_n_checkpoints(n: int, output_dir: str) -> str:
    """Усредняет последние n чекпоинтов и сохраняет в output_dir."""
    checkpoints = get_checkpoints(MODELS_DIR, n)
    avg_state_dict = average_checkpoints(checkpoints)

    os.makedirs(output_dir, exist_ok=True)
    last_checkpoint = checkpoints[-1]

    # Токенайзер
    tokenizer = AutoTokenizer.from_pretrained(last_checkpoint)
    tokenizer.save_pretrained(output_dir)

    # Config
    config_src = os.path.join(last_checkpoint, "config.json")
    config_dst = os.path.join(output_dir, "config.json")
    if os.path.exists(config_src):
        shutil.copy(config_src, config_dst)

    # Сохраняем веса
    weights_path = os.path.join(output_dir, "model.safetensors")
    safetensors.torch.save_file(avg_state_dict, weights_path)
    return output_dir


def evaluate_on_flores(model_dir: str) -> Tuple[float, float, int]:
    """Оценка только на FLORES-200 devtest EN→RU."""
    model, tokenizer = eval_mod.load_model(model_dir)
    sources, references = eval_mod.load_flores200_enru_devtest()
    hypotheses = eval_mod.translate_all(model, tokenizer, sources, batch_size=EVAL_BATCH_SIZE)
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    return round(bleu.score, 1), round(chrf.score, 1), len(sources)


def main():
    # Настраиваем eval.py глобально (используется в translate_all/translate_batch)
    eval_mod.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    eval_mod.NUM_BEAMS = NUM_BEAMS
    eval_mod.TEMPERATURE = TEMPERATURE
    eval_mod.MAX_INPUT_LENGTH = MAX_INPUT_LENGTH
    eval_mod.MAX_OUTPUT_LENGTH = MAX_OUTPUT_LENGTH
    eval_mod.FLORES200_URL = FLORES200_URL
    eval_mod.FLORES200_CACHE_DIR = FLORES200_CACHE_DIR
    eval_mod.FLORES200_SPLIT = FLORES200_SPLIT
    eval_mod.FLORES200_SRC_LANG = FLORES200_SRC_LANG
    eval_mod.FLORES200_TGT_LANG = FLORES200_TGT_LANG

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    results = []
    for n in range(RANGE_MAX, RANGE_MIN - 1, -1):
        out_dir = os.path.join(OUTPUT_ROOT, f"merged_{n}")
        print("\n" + "=" * 60)
        print(f"MERGE {n} чекпоинтов -> {out_dir}")
        print("=" * 60)
        merge_last_n_checkpoints(n, out_dir)
        bleu, chrf, count = evaluate_on_flores(out_dir)
        results.append({"n": n, "bleu": bleu, "chrf": chrf, "count": count})
        print(f">>> FLORES200 {FLORES200_SPLIT}: BLEU={bleu} | chrF={chrf} | n={count}")

    # Итоговая таблица
    print("\n" + "=" * 50)
    print(f"{'last_n':<10} {'BLEU':>8} {'chrF':>8} {'count':>8}")
    print("-" * 50)
    for r in results:
        print(f"{r['n']:<10} {r['bleu']:>8.1f} {r['chrf']:>8.1f} {r['count']:>8d}")
    print("=" * 50)


if __name__ == "__main__":
    main()
