# Day 15：翻訳→要約パイプライン

## 今日やったこと
- 翻訳→要約のエンドツーエンドパイプラインを実装
- max_position_embeddingsによる長文の制限を発見
- 文章を分割して翻訳する方法を実装
- BARTで英語要約を実装
- ドメイン特化モデルの重要性を再確認

---

## パイプラインの全体像

```
日本語テキスト
  ↓ 句点で分割（chunk_size=3文ずつ）
  ↓ Helsinki-NLP（opus-mt-ja-en）で翻訳
英語テキスト
  ↓ facebook/bart-large-cnn で要約
英語の要約文
```

---

## 実装

### 翻訳（Day 14から流用）

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name_ja_en = "Helsinki-NLP/opus-mt-ja-en"
tokenizer_ja_en = AutoTokenizer.from_pretrained(model_name_ja_en)
model_ja_en = AutoModelForSeq2SeqLM.from_pretrained(model_name_ja_en)

def translate_ja_en(text):
    inputs = tokenizer_ja_en(text, return_tensors="pt")
    outputs = model_ja_en.generate(**inputs)
    return tokenizer_ja_en.decode(outputs[0], skip_special_tokens=True)
```

### 長文分割翻訳

```python
def translate_ja_en_chunked(text, chunk_size=3):
    """文章を句点で分割して翻訳する"""
    sentences = [s.strip() for s in text.split('。') if s.strip()]
    translated_chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = '。'.join(sentences[i:i+chunk_size]) + '。'
        translated = translate_ja_en(chunk)
        translated_chunks.append(translated)
    return ' '.join(translated_chunks)
```

### 要約

```python
summarizer_name = "facebook/bart-large-cnn"
tokenizer_sum = AutoTokenizer.from_pretrained(summarizer_name)
model_sum = AutoModelForSeq2SeqLM.from_pretrained(summarizer_name)

def summarize(text, max_length=100, min_length=30):
    inputs = tokenizer_sum(text, return_tensors="pt", truncation=True)
    outputs = model_sum.generate(
        **inputs,
        max_length=max_length,
        min_length=min_length,
    )
    return tokenizer_sum.decode(outputs[0], skip_special_tokens=True)
```

---

## 発見した問題点と解決策

### max_position_embeddings による制限

```
MarianConfig の max_position_embeddings: 512
→ 入力・出力ともに512トークンが上限
→ 長い文章は途中で切れてしまう
```

解決策：句点で文章を分割してchunk_sizeずつ翻訳する。

### 専門用語の翻訳精度

```
教師あり学習 → Teacher learning    （正：Supervised learning）
教師なし学習 → Teacherless learning （正：Unsupervised learning）
強化学習    → Strong learning      （正：Reinforcement learning）
```

Helsinki-NLPは一般文章には強いが機械学習の専門用語は苦手。
→ ドメイン特化モデルが必要（Day 12で学んだ内容）

---

## キーワード集

### `**inputs`（辞書のアンパック）

```python
inputs = tokenizer(text, return_tensors="pt")
# inputs = {'input_ids': tensor(...), 'attention_mask': tensor(...)}

model.generate(**inputs)
# = model.generate(input_ids=tensor(...), attention_mask=tensor(...))
```

辞書をキーワード引数として展開して渡す。

### `attention_mask`

「どのトークンを注目してよいか」を示すマスク。

```
input_ids    ：[6124, 9988, 18, ...]  ← トークンID
attention_mask：[1,    1,   1,  ...]  ← 1=注目してよい・0=無視する
```

バッチ処理で文章の長さが違うとき、短い文章に追加される `[PAD]` トークンを無視するために使う。1文だけの場合は全部1になる。

### `truncation=True`

入力がモデルの最大長を超えた場合に自動で切り詰める。

### `max_position_embeddings`

モデルが処理できる最大トークン数。これを超えると情報が失われる。

---

## 今日の流れまとめ

```
1. 翻訳→要約のパイプラインを実装
2. max_position_embeddings=512 の制限で長文が途中で切れる問題を発見
3. 句点で分割してchunk_sizeずつ翻訳することで解決
4. 専門用語の翻訳精度が低い → ドメイン特化モデルの重要性を再確認
5. パイプライン全体の流れは正常に動作した
```

---

## 疑問・メモ欄
<!-- 読んでて気になったことをここに書く -->