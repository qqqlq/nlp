# Day 2：テキスト前処理 + Bag of Words + TF-IDF

## 今日やったこと
- なぜテキストをそのまま機械学習に使えないかを理解
- Bag of Words（BoW）でテキストを数値に変換
- TF-IDFでBoWの問題点を解決
- 日本語の分割問題に触れた

---

## なぜテキストはそのまま扱えないか

数値は大小・距離・演算が自然に定義できる（3は2より大きい、3-2=1）。  
テキストはそういった関係が定義されていない（「犬」と「猫」の差は？「好き」×2は？）。

→ **テキストを何らかのルールで数値に変換する必要がある**

---

## Bag of Words（BoW）

「文章を単語の出現回数の表に変換する」だけ。

```
「私は犬が好き」 → [1, 1, 1, 1, 1, 0, 0]  # 各単語の出現回数
「私は猫が好き」 → [1, 1, 0, 1, 1, 1, 0]
「犬も猫も好き」 → [0, 0, 1, 0, 1, 1, 2]
```

各行が「文章を数値化したベクトル」になる。

**問題点**：「は」「が」「も」のような助詞も頻出単語として同等に扱われてしまう。

---

## TF-IDF

BoWの問題点を解決する手法。

```
TF  = その単語がその文書に何回出てくるか（多いほど重要）
IDF = その単語が全文書のうち何割に出てくるか（少ない文書にしか出ない単語ほど重要）

TF-IDF = TF × IDF
```

→ **「その文書に多く出てくるが、他の文書にはあまり出てこない単語」を重要とみなす**

### 使い分けの目安

| 用途 | 手法 |
|------|------|
| とりあえず試したい・シンプルに | BoW |
| 文書の特徴を捉えたい・分類精度を上げたい | TF-IDF |
| 現代のNLP（BERTなど） | 単語埋め込み（Day5以降） |

---

## 日本語の分割問題

英語はスペースで自然に単語分割できる。  
日本語にはスペースがないので1文字ずつ分割すると意味が壊れる。

```python
# 英語：スペース区切りで自然に分割できる
corpus_en = ["I like dogs", "I like cats"]
vectorizer = CountVectorizer()  # デフォルトはスペース区切り

# 日本語：1文字ずつになってしまう
corpus_ja = ["私は犬が好き", "私は猫が好き"]
vectorizer = CountVectorizer(analyzer="char")  # 文字単位
# → 「好き」が「好」と「き」に分かれてしまう
```

→ 日本語を正しく単語分割するには**形態素解析**が必要（Day 3）

---

## キーワード集

### `CountVectorizer(analyzer=...)`
BoWの計算をする箱を作る処理。

```python
vectorizer = CountVectorizer(analyzer="char")
```

`analyzer` のオプション：
- `"word"`：スペースで単語分割（デフォルト）
- `"char"`：1文字ずつ分割

---

### `fit_transform(corpus)`
`fit` と `transform` を一度にやる処理。

```python
# この2行を1行でやってくれる
vectorizer.fit(corpus)        # 全文書を見て「単語一覧」を作る
X = vectorizer.transform(corpus)  # 各文書を数値ベクトルに変換する

# 1行で書くと
X = vectorizer.fit_transform(corpus)
```

`fit` だけだと単語一覧を作るだけで変換はしない。`transform` で初めて行列になる。

---

### `get_feature_names_out()`
`fit` で作った単語一覧を取り出す処理。

```python
vectorizer.get_feature_names_out()
# → ['and' 'cats' 'dogs' 'like']
```

行列のどの列がどの単語に対応するかを確認するために使う。

---

### 疎行列（sparse matrix）と `toarray()`

`fit_transform` の戻り値は疎行列という特殊な形式。

```
疎行列：0が多い行列をメモリ節約のために0以外の場所だけ記録する形式
(0, 4)  3  ← 0行目・4列目に3が入っている
(0, 5)  1  ← 0行目・5列目に1が入っている
```

文書が増えると単語一覧が膨大になりほとんどの値が0になるので、この形式が使われる。

```python
X.toarray()  # 普通のNumPy配列に変換（0も含めて全要素表示）
```

---

### `TfidfVectorizer()`
TF-IDFの計算をする箱。使い方は `CountVectorizer` と同じ。

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(corpus)
```

---

## 疑問・メモ欄
当たり前だがfitしたあとtransformせずとも単語一覧であれば出力可能