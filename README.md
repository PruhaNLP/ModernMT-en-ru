# PruhaNLP/ModernMT-en-ru-EXP — EN→RU переводчик на базе RuModernBERT

PruhaNLP/ModernMT-en-ru-EXP это экспериментальная модель для перевода с английского на русский (66M параметров). База — deepvk/RuModernBERT-small, инициализация декодера методом Bert2Bert ([статья](https://arxiv.org/abs/1907.12461)).

An experimental English→Russian translation model (66M params) based on deepvk/RuModernBERT-small. Decoder initialized via Bert2Bert method.

![FLORES-200 eval](pic/flores_200_eval.png)



## Использование модели

Предобученная модель: [PruhaNLP/ModernMT-en-ru-EXP](https://huggingface.co/PruhaNLP/ModernMT-en-ru-EXP).

```python
from transformers import AutoModel, AutoTokenizer

model_name = "PruhaNLP/ModernMT-en-ru-EXP"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
model.to("cuda").eval()

text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(text, return_tensors="pt").to("cuda")
output_ids = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=256,
    num_beams=4,
)
translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
```


## Evaluation

| Model | Params | FLORES-200 | WMT13 | WMT14 | WMT15 | WMT16 | WMT17 | WMT18 | WMT19 | WMT20 | WMT21 |
|:---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| facebook/wmt19-en-ru | ~300M | 30.4 | 29.7 | 43.1 | 40.3 | 35.8 | 42.2 | 34.9 | 33.4 | 23.8 | — |
| **PruhaNLP/ModernMT-en-ru-EXP** | **66M** | **29.5** | **24.8** | **38.9** | **32.0** | **30.1** | **33.9** | **29.9** | **29.8** | **23.2** | **25.3** |
| facebook/nllb-200-3.3B | 3.3B | 29.3 | 27.4 | 39.8 | 33.2 | 32.6 | 34.9 | 31.3 | 32.0 | 23.6 | 37.5 |
| facebook/nllb-200-distilled-1.3B | 1.3B | 28.5 | 27.4 | 39.5 | 33.5 | 32.8 | 34.8 | 31.7 | 32.2 | 23.6 | 37.3 |
| facebook/nllb-200-1.3B | 1.3B | 28.3 | 26.7 | 38.5 | 33.1 | 32.0 | 34.3 | 30.6 | 31.6 | 23.4 | 36.5 |
| facebook/m2m100_1.2B | 1.2B | 28.1 | 24.3 | 37.0 | 30.5 | 28.9 | 32.5 | 28.1 | 28.2 | 22.7 | — |
| gsarti/opus-mt-tc-base-en-ru | ~76M | 27.6 | 23.4 | 34.7 | 29.0 | 27.5 | 30.6 | 27.1 | 26.8 | 20.8 | — |
| facebook/nllb-200-distilled-600M | 600M | 25.6 | 25.0 | 35.4 | 29.9 | 29.1 | 31.4 | 27.8 | 29.1 | 21.6 | 32.7 |
| facebook/m2m100_418M | 418M | 22.5 | 20.5 | 30.4 | 25.6 | 24.0 | 26.4 | 22.7 | 23.4 | 18.6 | — |

## Обучение (детали)

Энкодер — RuModernBERT-small, декодер — каждым вторым слоем энкодера. Трапециевидный lr: 5% warmup, 20% decay. Merge последних 7 чекпойнтов.

Датасет: 100M пар из Helsinki-NLP/tatoeba_mt_train (eng-rus), фильтрация только по длине. Железо: 1× V100.

## Лицензия

Apache 2.0
