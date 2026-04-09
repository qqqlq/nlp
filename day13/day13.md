# Day 13：GPTシリーズの構造・テキスト生成の仕組み

## 今日やったこと
- BERTとGPTの構造的な違いを理解
- Causal Mask（因果マスク）の仕組みを理解
- GPT-2でテキスト生成を実装
- temperature・top_k・top_pの違いを理解

---

## BERTとGPTの違い

```
BERT（エンコーダ）：文章全体を同時に見る
  「私は[MASK]が好き」→ [MASK]に何が入るか予測
  → 理解タスクが得意（分類・感情分析など）

GPT（デコーダ）：左から右に順番に生成する
  「私は」→「犬」→「が」→「好き」
  → 生成タスクが得意（文章生成・翻訳など）
```

なぜGPTは左から右にしか生成できないか：
まだ生成していない未来の単語は存在しないので、既出の単語だけを参照して次を予測するしかない。

---

## Causal Mask（因果マスク）

BERTとGPTのAttentionの違い。

```
BERTのAttention：全単語が全単語を参照できる
GPTのAttention ：左側の単語しか参照できない（右側をマスク）

      私  は  犬  が  好き
私  [ ○   ×   ×   ×   × ]
は  [ ○   ○   ×   ×   × ]
犬  [ ○   ○   ○   ×   × ]
が  [ ○   ○   ○   ○   × ]
好き[ ○   ○   ○   ○   ○ ]
```

マスクの仕組み：参照してほしくない場所のスコアを `-∞` に置き換える。

```python
scores = scores + mask  # マスク（-∞）を足す
weights = softmax(scores)
# → -∞の部分は e^(-∞) = 0 になる → 注目度が完全に0
```

---

## テキスト生成の実装

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

result = generator(
    "The future of artificial intelligence",
    max_new_tokens=50,
)
print(result[0]['generated_text'])
```

---

## 生成パラメータ

次の1トークンを選ぶ流れ：

```
① スコア計算（logits）
      ↓
② temperature で割る（確率分布の尖り具合を調整）
      ↓
③ top_k or top_p で候補を絞る（明らかにおかしい単語を除外）
      ↓
④ 残った候補を再正規化
      ↓
⑤ 確率に従ってランダムに1つ選ぶ
      ↓
⑥ ①に戻って次のトークンを選ぶ → 繰り返す
```

### temperature

softmaxの前にスコアをtemperatureで割る。

```python
scaled = scores / temperature  # ← ここで割る
probs = softmax(scaled)
```

スコアが [3.0, 1.0, 0.5] のとき：

```
temperature=0.1：scaled=[30, 10, 5]  → probs=[1.000, 0.000, 0.000] ← ほぼ1択
temperature=1.0：scaled=[3, 1, 0.5] → probs=[0.821, 0.111, 0.067] ← 標準
temperature=1.5：scaled=[2, 0.67, 0.33] → probs=[0.688, 0.181, 0.130] ← ランダム
```

temperatureを下げると差が広がり1択に近くなる。上げると確率が均等に近づく。

### top_k

全語彙の中から確率上位k個だけ残して残りを除外し、再正規化してその中から1つ選ぶ。

```
全語彙の確率：
  "is"   : 0.60
  "will" : 0.20
  "can"  : 0.10
  "might": 0.05 ...

top_k=3 → is・will・can だけ残す → 再正規化 → 1つランダムに選ぶ
```

### top_p（nucleus sampling）

確率の高い順に足していって合計がpを超えるまでの候補だけ残して再正規化し1つ選ぶ。

```
"is"   : 0.60 → 合計0.60
"will" : 0.20 → 合計0.80
"can"  : 0.10 → 合計0.90 ← p=0.9を超えた
→ is・will・can だけ残す → 再正規化 → 1つランダムに選ぶ
```

top_pの利点：状況に応じて候補数が変わるので柔軟。

```
ほぼ1択の場合："the" : 0.95 → 合計0.95 > 0.9 → 1択になる（自然）
分散の場合   ：上位複数語で0.9に達する → 複数候補から選ぶ
```

現代のLLMではtop_pがよく使われる。

### do_sample

```
do_sample=False（デフォルト）：毎回確率が最も高いトークンを選ぶ → 毎回同じ文章
do_sample=True              ：確率に従ってランダムに選ぶ → 毎回違う文章
```

temperature・top_k・top_pを使うには `do_sample=True` が必要。

---

## キーワード集

### `set_seed(42)`
生成の再現性を確保する。

```python
from transformers import set_seed
set_seed(42)  # 毎回同じ出力になる
```

### `max_new_tokens`
新しく生成するトークンの最大数。

---

## 今日の流れまとめ

```
1. BERTはエンコーダ（理解）・GPTはデコーダ（生成）
2. Causal Maskで未来のトークンをマスク → 左から右に生成
3. マスク = -∞を足してsoftmax後に0にする
4. temperature・top_k・top_pで生成の多様性を制御
5. 全部「候補を絞って確率に従って1つ選ぶ」ための手法
```

---

## 疑問・メモ欄
<!-- 読んでて気になったことをここに書く -->