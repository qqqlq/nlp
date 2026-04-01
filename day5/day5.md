# Day 5：単語埋め込みの直感（Word2Vec・GloVe）

## 今日やったこと
- BoW・TF-IDFの限界を理解
- Word2Vecの仕組みと直感を理解
- コサイン類似度で単語の近さを計算
- PCAで300次元ベクトルを2次元に可視化

---

## BoW・TF-IDFの限界

```
「犬」と「猫」→ TF-IDFでは全く別のベクトル
「好き」と「大好き」→ TF-IDFでは全く別のベクトル
```

単語の**意味的な近さ**が全く表現できない。

---

## 単語埋め込みとは

各単語を数百次元のベクトルに変換する技術。

```
犬  → [0.2, -0.5, 0.8, 0.1, ...]  # 300次元
猫  → [0.3, -0.4, 0.7, 0.2, ...]  # 犬と似たベクトル
飛行機 → [-0.8, 0.9, -0.2, 0.6, ...]  # 犬と全然違うベクトル
```

---

## Word2Vecの核心

**「同じ文脈に出てくる単語は意味が近い」**という仮説に基づく。

```
「私は犬を飼っている」
「私は猫を飼っている」
→「犬」と「猫」は同じ文脈に出てくる → 意味が近い

「私は飛行機を飼っている」
→ 自然な文章では出てこない → 「犬」と「飛行機」は意味が遠い
```

大量のテキストからこのパターンを学習してベクトルを生成する。

---

## コサイン類似度

2つのベクトルの向きがどれだけ近いかを表す指標。-1〜1の値を取る。

```
1.0  → 完全に同じ向き（同じ意味）
0.0  → 直交（無関係）
-1.0 → 正反対の向き（反対の意味）
```

```python
def cosine_similarity(v1, v2):
    # 内積 ÷ (v1の長さ × v2の長さ)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
```

「ベクトルの長さを無視して向きだけで比較する」指標。

---

## ベクトル演算で意味の操作ができる

```python
# king - man + woman ≈ queen
model.most_similar(
    positive=["king", "woman"],  # 足す単語
    negative=["man"],            # 引く単語
    topn=3
)
# → queen: 0.7118

# Japan - Tokyo + Paris ≈ France
model.most_similar(
    positive=["Japan", "Paris"],
    negative=["Tokyo"],
    topn=3
)
# → France: 0.7890
```

ベクトル空間の中で「国と首都の関係」「性別の関係」などが表現されている。

---

## PCAで可視化

300次元のベクトルを2次元に圧縮して可視化する。

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)
```

動物グループ（dog・cat・puppy・kitten）と乗り物グループ（airplane・car・train・ship）が
2次元に圧縮しても別々のクラスタに分かれる
→ 元の300次元空間でも意味的なまとまりがある証拠。

次元削減の詳細は後日。

---

## キーワード集

### `KeyedVectors`（gensimのモデル構造）

辞書のように単語でアクセスできるオブジェクト。

```python
# 内部的にはこんな感じ（簡略化）
{
    "dog":      [0.05, -0.02, -0.17, ...],  # 300次元
    "cat":      [0.03, -0.04,  0.07, ...],  # 300次元
    ...  # 300万語分
}

# アクセス方法
model["dog"]          # → numpy配列 (300,)
model["dog"].shape    # → (300,)
```

---

### `most_similar`

意味的に近い単語を返す。戻り値は `(単語, コサイン類似度)` のタプルのリスト。

```python
model.most_similar("dog", topn=5)
# → [("dogs", 0.868), ("puppy", 0.810), ...]

# ベクトル演算
model.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=3
)
```

---

### アンパック

タプルや配列の要素を複数の変数に一度に代入する記法。

```python
# most_similarの戻り値
[("dogs", 0.868), ("puppy", 0.810), ...]

# アンパックで受け取る
for word, score in model.most_similar("dog", topn=5):
    print(word, score)

# 一般的なアンパック
a, b = (1, 2)         # a=1, b=2
x, y, z = [3, 4, 5]  # x=3, y=4, z=5
```

要素数と変数の数が一致していれば使える。

---

### `np.dot(v1, v2)`
2つのベクトルの内積を計算する。各次元を掛けて足し合わせる。

### `np.linalg.norm(v)`
ベクトルの長さ（ノルム）を計算する。

---

## 今日の流れまとめ

```
1. BoW・TF-IDFは意味的な近さを表現できない
2. Word2Vecは「同じ文脈に出る単語は意味が近い」で学習
3. コサイン類似度で単語の近さを数値化
4. ベクトル演算で意味の操作ができる（king - man + woman = queen）
5. PCAで300次元→2次元に圧縮して可視化 → グループ構造が確認できた
```

---

## 疑問・メモ欄
