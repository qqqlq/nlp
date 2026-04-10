# Day 14：翻訳モデルの実践（Helsinki-NLP）

## 今日やったこと
- エンコーダ＋デコーダ構造で翻訳が動く仕組みを理解
- Helsinki-NLPの翻訳モデルを実装
- BLEUスコアで翻訳精度を評価
- MarianMTモデルの使い方を学習

---

## 翻訳とTransformerの構造

```
エンコーダ：「私は犬が好き」を理解して内部表現に変換
デコーダ  ：内部表現から「I like dogs」を生成

BERTはエンコーダのみ → 理解タスク
GPTはデコーダのみ   → 生成タスク
翻訳はエンコーダ＋デコーダ両方が必要
```

---

## 実装

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "Helsinki-NLP/opus-mt-ja-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def translate(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result
```

### 翻訳結果
```
私は犬が好きです。         → I like dogs.
今日の天気はとても良いですね。 → The weather is very nice today, isn't it?
機械学習はとても面白い分野です。→ Machine learning is a very interesting field.
```

「とても良いですね」の「ね」（確認を求めるニュアンス）まで「isn't it?」と正確に翻訳している。

---

## 注意点

```
transformers v5 では pipeline("translation") が使えなくなった
→ AutoTokenizer + AutoModelForSeq2SeqLM で直接ロードする

Helsinki-NLP のモデルは MarianMT という専用アーキテクチャ
→ AutoTokenizer が内部で MarianTokenizer を自動選択してくれる
   （sentencepiece のインストールが必要）
```

---

## BLEUスコア

機械翻訳の精度を測る指標。「機械翻訳の結果が参照翻訳（人間が書いた正解）とどれだけ一致しているか」をn-gramの一致率で計算する。

```
0〜100のスコア
100：完全一致
0  ：全く一致しない

実際の目安：
  Google翻訳レベル：40〜60
  研究論文の最先端：50〜70
  人間の翻訳      ：〜100
```

### n-gramとは

連続するn個の単語がどれだけ一致しているかを測る。

```
参照：I like dogs
翻訳：I love dogs

1-gram（単語単位）  ：I・dogs が一致 → 2/3
2-gram（2単語単位） ：「I like」vs「I love」→ 不一致
3-gram（3単語単位） ：「I like dogs」vs「I love dogs」→ 不一致
```

単語の一致だけでなく「並び順」も考慮している。

### BLEUの計算式

BLEUはn=1〜4の4種類を全部計算して組み合わせる。

```
BLEU = BP × exp(0.25×log(p1) + 0.25×log(p2) + 0.25×log(p3) + 0.25×log(p4))

p1〜p4：1〜4gramの一致率（各0.25の重み）
BP    ：Brevity Penalty（翻訳が短すぎるとペナルティ）
```

単純な平均ではなく**幾何平均**。1-gramだけ高くても4-gramが低ければスコアは下がる。

### 今回の結果

```
完全一致（参照＝翻訳）：BLEU = 100.00
別の言い回しの参照   ：BLEU = 36.60
→「単語の一致はそこそこあるが語順や表現の一致が低い」状態
```

---

## キーワード集

### `AutoModelForSeq2SeqLM`
エンコーダ＋デコーダ構造のモデルをロードするクラス。翻訳・要約などのseq2seqタスクに使う。

### `model.generate(**inputs)`
モデルに入力を渡して出力トークン列を生成する。

### `tokenizer.decode(outputs[0], skip_special_tokens=True)`
生成されたトークンIDを文字列に変換する。`skip_special_tokens=True` で `[CLS]`・`[SEP]` などを除去。

### `sacrebleu`
BLEUスコアを計算する標準化されたライブラリ。論文間で比較可能な実装。

```python
import sacrebleu
bleu = sacrebleu.corpus_bleu(hypotheses, references)
print(bleu.score)
```

### モデルのキャッシュ
一度ダウンロードしたモデルは `~/.cache/huggingface/` に保存される。2回目以降はキャッシュから読み込むので高速。

---

## 今日の流れまとめ

```
1. 翻訳はエンコーダ＋デコーダ両方が必要
2. Helsinki-NLPのMarianMTモデルで日本語→英語翻訳を実装
3. transformers v5ではpipeline("translation")が使えない
   → AutoTokenizer + AutoModelForSeq2SeqLM で直接ロード
4. BLEUスコアで翻訳精度を評価（1〜4gramの幾何平均）
5. 完全一致で100点・別の言い回しの参照で36.60点
```

---

## 疑問・メモ欄

### 内部表現とは

エンコーダが入力文章を処理した後の「意味を数値化したベクトル」のこと。Day 9・10でやったAttentionの出力そのもの。

```
「私は犬が好き」
  ↓ エンコーダ（Transformerブロック × 複数層）
  ↓ 各トークンがAttentionで文脈を考慮したベクトルに変換される
内部表現：[[0.3, -0.5, 1.2, ...],   ← 「私」の文脈考慮済みベクトル
           [0.8,  0.2, -0.4, ...],   ← 「は」の文脈考慮済みベクトル
           [-0.1, 0.9,  0.7, ...],   ← 「犬」の文脈考慮済みベクトル
           ...]
```

この内部表現をデコーダに渡してCross-Attentionで参照しながら英語を生成する。

```
Self-Attention  ：自分の文章の中で単語同士が注目し合う（エンコーダ内）
Cross-Attention ：デコーダが生成中にエンコーダの内部表現を参照する
```

詳細は以下を参照：
- [Day 9：Attentionメカニズム](../day09/day09.md)
- [Day 10：Transformerの全体構造](../day10/day10.md)