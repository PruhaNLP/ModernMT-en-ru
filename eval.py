"""
Evaluation скрипт для EN→RU модели.
Бенчмарки: WMT newstest, Tatoeba
Метрики: BLEU, chrF
"""

import os
import glob
from typing import List, Tuple

import torch
from tqdm import tqdm
import sacrebleu
from transformers import AutoTokenizer, AutoModel

# ======Settings=========
MODELS_DIR = "merged_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 512
NUM_BEAMS = 4  # 1 = greedy, >1 = beam search
TEMPERATURE = 0.0  # 1.0 = без изменений, <1 = более уверенно, >1 = более разнообразно

# FLORES-200 devtest (EN→RU) — первый тест
FLORES200_ENABLED = True
FLORES200_URL = "https://tinyurl.com/flores200dataset"
FLORES200_CACHE_DIR = "data/flores200"  # сюда скачаем/распакуем
FLORES200_SPLIT = "devtest"
FLORES200_SRC_LANG = "eng_Latn"
FLORES200_TGT_LANG = "rus_Cyrl"

WMT_TESTSETS = [
    ("wmt13", "WMT13"),
    ("wmt14", "WMT14"),
    ("wmt15", "WMT15"),
    ("wmt16", "WMT16"),
    ("wmt17", "WMT17"),
    ("wmt18", "WMT18"),
    ("wmt19", "WMT19"),
    ("wmt20", "WMT20"),
    ("wmt21", "WMT21"),
]
# ======Settings=========


def get_latest_checkpoint(models_dir: str) -> str:
    checkpoints = glob.glob(os.path.join(models_dir, "checkpoint-*"))
    if not checkpoints:
        if os.path.exists(os.path.join(models_dir, "config.json")):
            return models_dir
        raise FileNotFoundError(f"Не найдено чекпойнтов в {models_dir}")
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return checkpoints[-1]


def load_model(checkpoint_path: str):
    from train import (
        EncoderDecoderModel,
        EncoderDecoderConfig,
        MODEL_NAME,
    )
    
    print(f"Загрузка модели из: {checkpoint_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    config = EncoderDecoderConfig.from_pretrained(checkpoint_path)
    
    model = EncoderDecoderModel(config)
    
    # Устанавливаем encoder
    encoder = AutoModel.from_pretrained(MODEL_NAME)
    
    # Resize embeddings если vocab_size изменился
    if encoder.embeddings.tok_embeddings.weight.shape[0] != config.vocab_size:
        print(f"Resizing encoder embeddings: {encoder.embeddings.tok_embeddings.weight.shape[0]} → {config.vocab_size}")
        encoder.resize_token_embeddings(config.vocab_size)
    
    model.set_encoder(encoder)
    
    # Загружаем веса
    import safetensors.torch
    weights_path = os.path.join(checkpoint_path, "model.safetensors")
    if os.path.exists(weights_path):
        state_dict = safetensors.torch.load_file(weights_path)
    else:
        weights_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location="cpu")
    
    model.load_state_dict(state_dict, strict=False)
    model.tie_weights()
    
    # Отключаем compile для encoder
    if hasattr(model.encoder, 'config'):
        model.encoder.config.reference_compile = False
    
    model.to(DEVICE)
    model.eval()
    
    return model, tokenizer


def _download_file(url: str, dst_path: str):
    import requests
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _ensure_flores200_dataset(cache_dir: str) -> str:
    """
    Качает/распаковывает FLORES-200 в cache_dir.
    Возвращает путь до папки, внутри которой лежит `flores200_dataset/`.
    """
    expected = os.path.join(
        cache_dir,
        "flores200_dataset",
        FLORES200_SPLIT,
        f"{FLORES200_SRC_LANG}.{FLORES200_SPLIT}",
    )
    if os.path.exists(expected):
        return cache_dir

    os.makedirs(cache_dir, exist_ok=True)
    archive_path = os.path.join(cache_dir, "flores200_dataset.tar.gz")
    if not os.path.exists(archive_path):
        print(f"[FLORES] Скачиваем FLORES-200: {FLORES200_URL}")
        _download_file(FLORES200_URL, archive_path)

    print(f"[FLORES] Распаковываем: {archive_path}")
    import tarfile
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=cache_dir)

    if not os.path.exists(expected):
        raise FileNotFoundError(f"[FLORES] Не найдено после распаковки: {expected}")
    return cache_dir


def load_flores200_enru_devtest() -> Tuple[List[str], List[str]]:
    """
    FLORES-200 devtest EN→RU.
    Используем файловую структуру как в `flores.py`:
    flores200_dataset/{split}/{lang}.{split}
    """
    try:
        root = _ensure_flores200_dataset(FLORES200_CACHE_DIR)
        src_path = os.path.join(root, "flores200_dataset", FLORES200_SPLIT, f"{FLORES200_SRC_LANG}.{FLORES200_SPLIT}")
        tgt_path = os.path.join(root, "flores200_dataset", FLORES200_SPLIT, f"{FLORES200_TGT_LANG}.{FLORES200_SPLIT}")

        with open(src_path, "r", encoding="utf-8") as f:
            sources = [line.strip() for line in f if line.strip()]
        with open(tgt_path, "r", encoding="utf-8") as f:
            references = [line.strip() for line in f if line.strip()]

        n = min(len(sources), len(references))
        sources, references = sources[:n], references[:n]
        return sources, references
    except Exception as e:
        print(f"Ошибка загрузки FLORES-200: {e}")
        return [], []


def beam_search_decode(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_length: int,
    eos_token_id: int,
    decoder_start_token_id: int,
    num_beams: int = 5,
) -> torch.Tensor:
    """Batched beam search decoding."""
    batch_size = input_ids.shape[0]
    device = input_ids.device
    vocab_size = model.config.vocab_size
    temperature = 1.0 if TEMPERATURE == 0 else TEMPERATURE
    
    # Расширяем inputs для всех beams: (batch, seq) -> (batch * num_beams, seq)
    expanded_input_ids = input_ids.unsqueeze(1).expand(-1, num_beams, -1).reshape(batch_size * num_beams, -1)
    expanded_attention_mask = attention_mask.unsqueeze(1).expand(-1, num_beams, -1).reshape(batch_size * num_beams, -1)
    
    # Decoder input: (batch * num_beams, 1)
    decoder_input_ids = torch.full((batch_size * num_beams, 1), decoder_start_token_id, dtype=torch.long, device=device)
    
    # Scores для каждого beam: (batch, num_beams)
    beam_scores = torch.zeros(batch_size, num_beams, device=device)
    beam_scores[:, 1:] = -1e9  # Только первый beam активен изначально
    beam_scores = beam_scores.view(-1)  # (batch * num_beams,)
    
    for step in range(max_length - 1):
        outputs = model(
            input_ids=expanded_input_ids,
            attention_mask=expanded_attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        
        # (batch * num_beams, vocab)
        next_token_logits = outputs.logits[:, -1, :]
        finished = decoder_input_ids[:, -1] == eos_token_id
        if finished.any():
            next_token_logits = next_token_logits.clone()
            next_token_logits[finished] = float("-inf")
            next_token_logits[finished, eos_token_id] = 0.0
        # TEMPERATURE=0 трактуем как "без температурного масштабирования"
        next_token_scores = torch.log_softmax(next_token_logits / temperature, dim=-1)
        
        # Добавляем накопленные scores: (batch * num_beams, vocab)
        next_token_scores = next_token_scores + beam_scores.unsqueeze(-1)
        
        # Reshape для выбора лучших: (batch, num_beams * vocab)
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
        
        # Top num_beams кандидатов
        next_scores, next_tokens = next_token_scores.topk(num_beams, dim=-1)
        
        # Определяем beam index и token id
        next_beam_indices = next_tokens // vocab_size  # (batch, num_beams)
        next_tokens = next_tokens % vocab_size  # (batch, num_beams)
        
        # Выбираем предыдущие последовательности по индексам beams
        batch_offset = (torch.arange(batch_size, device=device) * num_beams).unsqueeze(1)
        next_beam_indices = (next_beam_indices + batch_offset).view(-1)
        decoder_input_ids = decoder_input_ids[next_beam_indices]
        decoder_input_ids = torch.cat([decoder_input_ids, next_tokens.view(-1, 1)], dim=1)
        beam_scores = next_scores.view(-1)

        # Останавливаемся, когда у КАЖДОГО примера все beams закончились EOS
        last_tokens = decoder_input_ids.view(batch_size, num_beams, -1)[:, :, -1]
        done = (last_tokens == eos_token_id).all(dim=1)
        if done.all():
            break
    
    # Берём лучший beam для каждого примера по score
    final_scores = beam_scores.view(batch_size, num_beams)
    best_beam_idx = final_scores.argmax(dim=1)  # (batch,)
    sequences = decoder_input_ids.view(batch_size, num_beams, -1)
    best_sequences = sequences[torch.arange(batch_size, device=device), best_beam_idx, :]
    
    return best_sequences


def greedy_decode(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_length: int,
    eos_token_id: int,
    decoder_start_token_id: int,
) -> torch.Tensor:
    """Sampling decoding с температурой (TEMPERATURE=0 → greedy argmax)."""
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
        
        if TEMPERATURE == 0:
            # Greedy (argmax)
            next_tokens = next_token_logits.argmax(dim=-1)
        else:
            # Sampling с температурой
            probs = torch.softmax(next_token_logits / TEMPERATURE, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        next_tokens = next_tokens.masked_fill(finished, decoder_start_token_id)

        decoder_input_ids = torch.cat([decoder_input_ids, next_tokens.unsqueeze(-1)], dim=-1)

        finished = finished | (next_tokens == eos_token_id)
        if finished.all():
            break

    return decoder_input_ids


def translate_batch(model, tokenizer, texts: List[str], debug: bool = False, use_beam: bool = True) -> List[str]:
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    ).to(DEVICE)
    
    eos_token_id = model.config.eos_token_id
    decoder_start_token_id = model.config.decoder_start_token_id
    
    use_amp = (DEVICE == "cuda")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        if use_beam and NUM_BEAMS > 1:
            output_ids = beam_search_decode(
                model,
                inputs["input_ids"],
                inputs["attention_mask"],
                max_length=MAX_OUTPUT_LENGTH,
                eos_token_id=eos_token_id,
                decoder_start_token_id=decoder_start_token_id,
                num_beams=NUM_BEAMS,
            )
        else:
            output_ids = greedy_decode(
                model,
                inputs["input_ids"],
                inputs["attention_mask"],
                max_length=MAX_OUTPUT_LENGTH,
                eos_token_id=eos_token_id,
                decoder_start_token_id=decoder_start_token_id,
            )
    
    if debug:
        print(f"[DEBUG] Input IDs: {inputs['input_ids'].tolist()}")
        print(f"[DEBUG] Output IDs: {output_ids.tolist()}")
        print(f"[DEBUG] Output tokens: {[tokenizer.convert_ids_to_tokens(ids) for ids in output_ids.tolist()]}")
    
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)


def translate_all(model, tokenizer, texts: List[str], batch_size: int = BATCH_SIZE) -> List[str]:
    all_translations = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
        batch = texts[i:i + batch_size]
        translations = translate_batch(model, tokenizer, batch)
        all_translations.extend(translations)
    return all_translations


def check_testset_available(testset_id: str) -> bool:
    """Проверяет доступность тестсета."""
    for langpair in ["en-ru", "ru-en"]:
        try:
            sacrebleu.get_source_file(testset_id, langpair)
            return True
        except:
            continue
    return False


def load_wmt_testset(testset_id: str) -> Tuple[List[str], List[str]]:
    try:
        for langpair in ["en-ru", "ru-en"]:
            try:
                sources = sacrebleu.get_source_file(testset_id, langpair)
                references = sacrebleu.get_reference_files(testset_id, langpair)[0]
                
                with open(sources, "r", encoding="utf-8") as f:
                    src_lines = [line.strip() for line in f]
                with open(references, "r", encoding="utf-8") as f:
                    ref_lines = [line.strip() for line in f]
                
                if langpair == "ru-en":
                    src_lines, ref_lines = ref_lines, src_lines
                
                return src_lines, ref_lines
            except:
                continue
        raise ValueError(f"Не найден testset {testset_id}")
    except Exception as e:
        print(f"Ошибка загрузки {testset_id}: {e}")
        return [], []


def load_tatoeba() -> Tuple[List[str], List[str]]:
    try:
        from datasets import load_dataset
        ds = load_dataset("tatoeba", lang1="en", lang2="ru", split="train")
        sources = [item["translation"]["en"] for item in ds]
        references = [item["translation"]["ru"] for item in ds]
        return sources, references
    except Exception as e:
        print(f"Ошибка загрузки Tatoeba: {e}")
        return [], []


def evaluate_testset(model, tokenizer, sources: List[str], references: List[str], name: str) -> dict:
    if not sources:
        return {"name": name, "bleu": None, "chrf": None, "error": "Failed to load"}
    
    print(f"\nОценка на {name} ({len(sources)} примеров)...")
    
    hypotheses = translate_all(model, tokenizer, sources)
    
    print("\n--- Примеры переводов ---")
    for i in range(min(5, len(sources))):
        print(f"SRC: {sources[i][:80]}...")
        print(f"REF: {references[i][:80]}...")
        print(f"HYP: {hypotheses[i][:80] if hypotheses[i] else '[ПУСТО]'}...")
        print()
    
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    
    print(f"\n>>> {name}: BLEU = {bleu.score:.1f} | chrF = {chrf.score:.1f}")
    
    return {
        "name": name,
        "bleu": round(bleu.score, 1),
        "chrf": round(chrf.score, 1),
        "count": len(sources),
    }


def print_results_table(results: List[dict]):
    print("\n" + "=" * 50)
    print(f"{'testset':<25} {'BLEU':>8} {'chrF':>8}")
    print("-" * 50)
    
    for r in results:
        if r.get("error"):
            print(f"{r['name']:<25} {'ERROR':>8} {'ERROR':>8}")
        else:
            bleu_str = f"{r['bleu']:.1f}" if r['bleu'] is not None else "N/A"
            chrf_str = f"{r['chrf']:.1f}" if r['chrf'] is not None else "N/A"
            print(f"{r['name']:<25} {bleu_str:>8} {chrf_str:>8}")
    
    print("=" * 50)


def main():
    checkpoint_path = get_latest_checkpoint(MODELS_DIR)
    print(f"Используем чекпойнт: {checkpoint_path}")
    
    model, tokenizer = load_model(checkpoint_path)
    
    # Тестовые переводы
    test_sentences = [
        # 1. Многозначность (Context)
        "I need to watch some movie.",
        # 2. Фразовый глагол (Phrasal Verb)
        "She decided to turn down the job offer because the salary was too low.",
        # 3. Условное наклонение (Conditionals)
        "If I had known about the traffic, I would have left home earlier.",
        # 4. Разговорный стиль (Casual)
        "Honestly, I'm not a big fan of horror movies, they scare me to death.",
        # 5. Структура предложения (Relative Clause)
        "The book that you recommended to me last week was absolutely fascinating.",
        # 6. Бизнес/IT лексика
        "Please update the settings to ensure that the user data is secure.",
        # 7. Идиоматическое выражение (Lite)
        "It was a piece of cake for him to fix the broken chair.",
        # 8. Пассивный залог (Passive Voice)
        "The decision was made by the committee after a long discussion.",
        # 9. Абстракция (Descriptive)
        "A deep silence filled the room as he waited for the answer.",
        # 10. Отрицание с нюансом
        "I haven't seen him since we graduated from university.",
    ]
    
    print(f"\n{'='*60}")
    print("ТЕСТОВЫЕ ПЕРЕВОДЫ")
    print('='*60)
    
    for test_src in test_sentences:
        translation = translate_batch(model, tokenizer, [test_src], debug=False)
        print(f"EN: {test_src}")
        print(f"RU: {translation[0]}")
        print("-" * 40)
    
    # Один перевод с debug
    print(f"\n{'='*60}")
    print("DEBUG перевод:")
    print('='*60)
    test_translation = translate_batch(model, tokenizer, [test_sentences[0]], debug=True)
    print(f"Output: {test_translation[0]}")
    print("=" * 60 + "\n")
    
    # Проверяем доступность тестсетов
    print(f"\n{'='*60}")
    print("ПРОВЕРКА ДОСТУПНОСТИ ТЕСТСЕТОВ")
    print('='*60)
    available_testsets = []
    for testset_id, display_name in WMT_TESTSETS:
        if check_testset_available(testset_id):
            print(f"  ✓ {display_name}")
            available_testsets.append((testset_id, display_name))
        else:
            print(f"  ✗ {display_name} (недоступен)")
    print('='*60)
    
    results = []

    # Первый тест: FLORES-200 devtest en→ru
    if FLORES200_ENABLED:
        flores_sources, flores_references = load_flores200_enru_devtest()
        flores_name = f"FLORES200.{FLORES200_SPLIT}.{FLORES200_SRC_LANG}-{FLORES200_TGT_LANG}"
        result = evaluate_testset(model, tokenizer, flores_sources, flores_references, flores_name)
        results.append(result)
    
    for testset_id, display_name in available_testsets:
        sources, references = load_wmt_testset(testset_id)
        result = evaluate_testset(model, tokenizer, sources, references, display_name)
        results.append(result)
    
    sources, references = load_tatoeba()
    result = evaluate_testset(model, tokenizer, sources, references, "Tatoeba.en.ru")
    results.append(result)
    
    print_results_table(results)


if __name__ == "__main__":
    main()
