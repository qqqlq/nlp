# Day 11：BERT入門・HuggingFaceの使い方

## 今日やったこと
- 事前学習・ファインチューニングの概念を理解
- HuggingFaceの `pipeline` で感情分析を5行で実装
- トークナイザー（WordPiece）の仕組みを理解
- `pipeline` の内部処理を自分で再現

---

## 事前学習・ファインチューニング

```
事前学習（Pre-training）：
  大量のテキスト（Wikipedia等）でBERTを学習済みにする
  → Googleが実施済み → 誰でも使える
  コスト：数億円・数週間〜数ヶ月

ファインチューニング（Fine-tuning）：
  学習済みBERTを自分のタスクに合わせて追加学習する
  → 少ないデータ・短い時間で高精度が出る
```

料理に例えると：
```
事前学習       ：料理の基礎を習得済みのシェフ
ファインチューニング：そのシェフに特定の料理を追加で教える
```

---

## 使ったモデル：DistilBERT

```
distilbert-base-uncased-finetuned-sst-2-english

distilbert    ← BERTを軽量化（約60%のサイズ）
base          ← ベースサイズ
uncased       ← 大文字小文字を区別しない（I→i）
finetuned-sst-2 ← SST-2（映画レビューデータセット）でファインチューニング済み
english       ← 英語専用
```

### モデルの構造（config から確認）
```
dim: 768          ← 各トークンのベクトル次元数（d_model）
n_heads: 12       ← Multi-Head Attentionのヘッド数
n_layers: 6       ← Transformerブロックの層数
hidden_dim: 3072  ← FFNの中間層の次元数（768×4）
vocab_size: 30522 ← 語彙数
max_position_embeddings: 512 ← 最大512トークンまで処理可能
```

---

## pipelineを使った感情分析（5行）

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love this movie! It was fantastic.")
# → [{'label': 'POSITIVE', 'score': 0.9999}]
```

### pipelineの内部処理

```
① tokenizer(text)        ← テキストをトークンIDに変換
② model(**inputs)        ← BERTに通してlogitsを取得
③ softmax(logits)        ← 確率に変換
④ argmax → ラベル変換    ← 最大確率のラベルを返す
```

---

## トークナイザー（WordPiece）

BERTはテキストをそのまま受け取らず、まずトークンに分割する。

```python
tokenizer.tokenize("I love dogs and cats!")
# → ['i', 'love', 'dogs', 'and', 'cats', '!']

tokenizer.tokenize("unbelievably")
# → ['un', '##bel', '##ie', '##va', '##bly']
```

### 特殊トークン
```
[CLS]：文の先頭。分類タスクでは[CLS]の出力が「文章全体の意味」を表す
[SEP]：文の末尾。複数の文を区切るときにも使う
```

### WordPieceの仕組み
頻出単語はそのまま、珍しい単語はサブワードに分割する。

```
plane        → ['plane']       ← 頻出単語なのでそのまま
unbelievably → ['un', '##bel', '##ie', '##va', '##bly']  ← 珍しいので分割
unbelievable → ['unbelievable'] ← 映画レビューで頻出なのでそのまま
```

`##` は「前のトークンの続き」という意味。意味的な分割ではなく統計的な頻出パターンで分割。

メリット：語彙数を抑えながら未知語を減らせる。造語や専門用語にも対応できる。

---

## pipelineを使わずに内部を再現

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# ① トークン化
text = "I love dogs and cats!"
inputs = tokenizer(text, return_tensors="pt")  # PyTorchテンソルで返す
# inputs['input_ids'].shape → (1, 8)  バッチサイズ1 × トークン数8

# ② モデルに通す
with torch.no_grad():
    outputs = model(**inputs)

# ③ logitsをsoftmaxで確率に変換
probs = torch.softmax(outputs.logits, dim=-1)
print("NEGATIVE:", probs[0][0].item())  # 0.00022
print("POSITIVE:", probs[0][1].item())  # 0.99978
```

---

## キーワード集

### `pipeline(task)`
HuggingFaceの高レベルAPI。タスク名を渡すだけで適切なモデルをロードして使える。

```python
classifier = pipeline("sentiment-analysis")   # 感情分析
translator  = pipeline("translation")          # 翻訳
generator   = pipeline("text-generation")      # テキスト生成
```

### `AutoTokenizer.from_pretrained(model_name)`
モデル名からトークナイザーをロードする。

### `AutoModelForSequenceClassification.from_pretrained(model_name)`
モデル名からシーケンス分類モデルをロードする。

### `return_tensors="pt"`
トークナイザーの出力形式を指定する。
```
"pt" → PyTorch tensor
"np" → NumPy array
"tf" → TensorFlow tensor
```

### `logits`
softmaxを適用する前の生のスコア。確率ではなく「どちらのクラスにどれだけ引っ張られているか」を表す。

```python
logits: [-4.04, 4.37]  # NEGATIVE, POSITIVE
softmax後：[0.00022, 0.99978]
```

### `.item()`
テンソルからPythonのネイティブな値（int・float）に変換する。

```python
t = torch.tensor(0.9998)
t          # → tensor(0.9998)  ← テンソルのまま
t.item()   # → 0.9998          ← Pythonのfloatとして取り出す
```

### `zip(list1, list2)`
複数のリストを同時に1つずつ取り出す。

```python
for text, result in zip(texts, results):
    print(text, result)
```

### `model(**inputs)`
辞書をアンパックしてキーワード引数として渡す。

```python
inputs = {'input_ids': ..., 'attention_mask': ...}
model(**inputs)
# model(input_ids=..., attention_mask=...) と同じ
```

---

## 今日の流れまとめ

```
1. 事前学習済みモデルを使うことで少ないコストで高精度が出る
2. pipeline で5行で感情分析が動いた
3. WordPieceで頻出単語はそのまま・珍しい単語はサブワードに分割
4. [CLS][SEP]という特殊トークンが文の構造を示す
5. pipelineの内部はtokenizer→model→softmax→ラベル変換の4ステップ
```

---

## 疑問・メモ欄

### HuggingFaceとは何か

```
BERT開発：Google
GPT開発 ：OpenAI
LLaMA  ：Meta

HuggingFace：これらのモデルを誰でも使えるように
             ライブラリ（transformers）とプラットフォームを提供
```

2つの役割：
```
① transformers ライブラリ：BERTやGPTを簡単に使えるPythonライブラリ
② Model Hub：学習済みモデルを公開・共有するプラットフォーム（GitHubのモデル版）
```

`from_pretrained("モデル名")` の一行でModel Hubからモデルをダウンロードして使える。

最近は自社でも BLOOM・Falcon などのモデルを開発・公開している。  
「AI界のGitHub兼研究機関」という立ち位置。