# Day 3：日本語NLPの特殊性 + 形態素解析

## 今日やったこと
- 形態素・形態素解析の概念を理解
- janomeで日本語テキストを形態素解析
- 品詞フィルタリングと原形への統一
- 形態素解析 + TF-IDFを組み合わせた前処理パイプラインを実装

---

## 形態素とは

「意味を持つ最小の言語単位」。

```
「私は犬が好きだ」→ 私 / は / 犬 / が / 好き / だ
```

「好き」はこれ以上分割すると意味が壊れるので1つの形態素。

### なぜ日本語は難しいか

```
英語：スペースで自然に単語分割できる
      "I like dogs" → I / like / dogs

日本語：スペースがないのでどこで切るかわからない
        「私は犬が好きだ」→ そのままでは分割できない
```

→ **形態素解析**で解決する

---

## janomeの使い方

```python
from janome.tokenizer import Tokenizer

tokenizer = Tokenizer()
tokens = tokenizer.tokenize("私は犬が好きだ")

for token in tokens:
    print(token)
```

出力の構造：
```
表層形  品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音

私      名詞,代名詞,一般,*,*,*,私,ワタシ,ワタシ
は      助詞,係助詞,*,*,*,*,は,ハ,ワ
犬      名詞,一般,*,*,*,*,犬,イヌ,イヌ
好き    名詞,形容動詞語幹,*,*,*,*,好き,スキ,スキ
```

`*` は「該当なし」。名詞には活用がないので `*` になる。

### NLPで重要な3つの情報

```python
token.surface           # 表層形（文章中の実際の形）
token.part_of_speech    # 品詞情報（カンマ区切り）
token.base_form         # 原形
```

品詞だけ取り出すには：
```python
pos = token.part_of_speech.split(',')[0]  # カンマ区切りの最初の要素
```

---

## 原形への統一

活用形を原形に統一することで、同じ単語が別々にカウントされるのを防ぐ。

```
走る・走っ・走り・走れ → すべて「走る」に統一
好きだ・好きな・好きで → すべて「好き」に統一
```

---

## 品詞フィルタリング

助詞・助動詞（「は」「が」「て」「だ」など）は文章の意味を特徴づけないので除外する。

```python
def extract_nouns_verbs(text):
    """名詞と動詞だけ原形で取り出す関数"""
    tokens = tokenizer.tokenize(text)
    result = []
    for token in tokens:
        pos = token.part_of_speech.split(',')[0]
        if pos in ['名詞', '動詞']:
            result.append(token.base_form)
    return result  # returnはforループの外に置く！
```

よくあるミス：`return` をforループの中に書くと最初のトークンで即returnしてしまう。

---

## 形態素解析 + TF-IDFパイプライン

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "私は犬が好きだ",
    "猫も犬も好きだ",
    "犬を飼いたいと思っている",
]

# 各文章を形態素解析してスペース区切りの文字列に変換
processed = []
for text in corpus:
    tokens = extract_nouns_verbs(text)
    processed.append(' '.join(tokens))  # リストをスペース区切りの文字列に変換

# TF-IDFベクトル化（1文字も含める）
tfidf = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
X = tfidf.fit_transform(processed)
```

---

## キーワード集

### `' '.join(tokens)`
リストの要素を指定した文字列でつなげて1つの文字列にする処理。

```python
tokens = ['私', '犬', '好き']
' '.join(tokens)   # → '私 犬 好き'
','.join(tokens)   # → '私,犬,好き'
''.join(tokens)    # → '私犬好き'
```

TfidfVectorizer がスペース区切りの文字列を期待するので `' '` を使う。

---

### `token_pattern`
TfidfVectorizer が「どんな文字列を単語として認識するか」を正規表現で指定するオプション。

```python
# デフォルト：2文字以上の単語のみ（1文字の「犬」「猫」が除外される）
tfidf = TfidfVectorizer()

# 1文字以上の単語を認識する
tfidf = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
```

正規表現は「文字列のパターンを表す記法」。詳細は後日。

---

## TF-IDFの復習（重要な勘違いポイント）

```
TF  → 1つの文書の中での話（その文書で何回出るか）
IDF → 全文書をまたいだ話（何割の文書に出るか）
```

「犬」が全文書に登場するとIDFが低くなりTF-IDFスコアも低くなる。  
「私」が1文書にしか登場しないとIDFが高くなりTF-IDFスコアも高くなる。

→ **「その文書でよく出て、かつ他の文書では珍しい単語」が高スコア**

---

## 疑問・メモ欄

### `token_pattern` の正規表現について

正規表現とは「こういうパターンの文字列を探して」という指示を記号で書く記法。

```python
r'(?u)\b\w+\b'
#  ↑    ↑  ↑
#  ①    ②  ③
```

```
① (?u)  Unicodeモード（日本語などを正しく扱うための指定）
② \w+   「単語を構成する文字」が1文字以上続く
          \w = 英数字・ひらがな・カタカナ・漢字など
          +  = 1文字以上
③ \b   単語の境界（単語の始まりと終わりを示す）
```

デフォルトの `r'(?u)\b\w\w+\b'` との違いは `\w\w+`（2文字以上）か `\w+`（1文字以上）かだけ。  
正規表現はNLPの前処理でURLや記号を除去するときなどにも使う。毎回調べながら使うものなので「こういう記法がある」程度で十分。

---

### 形態素解析後にTF-IDFを使うとき、どの品詞を取るべきか？

**タスクによるが、一般的な目安：**

| タスク | 取る品詞 |
|--------|---------|
| 文書分類・感情分析 | 名詞・動詞・形容詞 |
| トピック抽出 | 名詞のみ |
| 検索・類似度計算 | 名詞・動詞・形容詞・副詞 |

**各品詞の役割：**

```
名詞：話題のテーマを表す → ほぼ必ず取る
動詞：行動・状態を表す → よく取る
形容詞：感情分析では特に重要（「美しい」「楽しい」）
副詞：強調・否定を表す → 感情分析では有用なことも
      「とても面白い」と「全然面白くない」の強度の差が消えてしまうため
助詞・助動詞：ほぼ常に除外
```

**実用的な出発点：** 名詞・動詞・形容詞の3つから始めて精度を見ながら調整する。

```python
def extract_words(text):
    tokens = tokenizer.tokenize(text)
    result = []
    for token in tokens:
        pos = token.part_of_speech.split(',')[0]
        if pos in ['名詞', '動詞', '形容詞']:
            result.append(token.base_form)
    return result
```