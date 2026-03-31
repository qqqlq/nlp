# Day 4：ロジスティック回帰でテキスト分類・評価指標

## 今日やったこと
- ロジスティック回帰の仕組みを理解
- シグモイド関数を実装・可視化
- IMDbレビューデータセットで感情分析（88%の精度）
- 評価指標（accuracy・precision・recall・F1）を理解

---

## 線形回帰 vs ロジスティック回帰

```
線形回帰       ：-∞ ～ +∞ の実数を予測（売上がいくらか）
ロジスティック回帰：0 ～ 1 の確率を予測（スパムである確率は何%か）
```

出力が確率なので「0.5以上ならクラス1」のように閾値で分類できる。

---

## シグモイド関数

どんな値でも0〜1に押し込める関数。

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

```
x が非常に小さい → 0に近づく（確率ほぼ0）
x = 0           → 0.5（境界）
x が非常に大きい → 1に近づく（確率ほぼ1）
```

xが0から離れると変化が小さくなる → **飽和**と呼ぶ。  
後日ニューラルネットの文脈で問題になる。

---

## 評価指標

スパム分類を例に：

```
accuracy（精度）  ：全体のうち正解した割合
                   → シンプルだがクラスが偏っていると意味をなさない

precision（適合率）：「スパムと予測したもののうち本当にスパムだった割合」
                   → 誤検知を減らしたいときに重視

recall（再現率）  ：「本当にスパムのもののうち正しく検出できた割合」
                   → 見逃しを減らしたいときに重視

f1-score         ：precisionとrecallの調和平均
                   → どちらも重視したいときに使う
```

---

## キーワード集

### `train_test_split(X, y, test_size=0.3, random_state=42)`
データを学習用とテスト用に分割する処理。

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42
)
```

```
test_size=0.3   → 30%をテスト用、70%を学習用にする
random_state=42 → 分割のランダムさを固定する（再現性のため）
```

学習データで学習・テストデータで評価する理由：  
学習に使ったデータで評価すると「答えを見てテストする」状態になるため。

---

### `random_state` と `np.random.seed()` の関係

同じ概念。どちらも「乱数の出発点を固定する」処理。

```python
np.random.seed(42)          # NumPyの乱数を固定
train_test_split(..., random_state=42)  # scikit-learnの乱数を固定
```

`42` 自体に意味はなく慣習的によく使われているだけ。`0` でも `123` でも動作は同じ。

---

### `load_files` が返す Bunch オブジェクトの構造

```
aclImdb/train/
    pos/  ← フォルダ名がラベルになる
        1.txt
        2.txt
    neg/  ← フォルダ名がラベルになる
        1.txt
        2.txt
```

```python
train_data = load_files("aclImdb/train", categories=["pos", "neg"],
                        encoding="utf-8")

train_data.data         # 各txtファイルのテキスト内容のリスト
train_data.filenames    # 各txtファイルのパス
train_data.target_names # フォルダ名のリスト（アルファベット順）
train_data.target       # 各ファイルがどのフォルダにあったか（数値）
train_data.DESCR        # データセットの説明（ない場合はNone）
```

ラベルの割り当ては**アルファベット順**：
```
neg → 0  （nが先）
pos → 1  （pが後）
```

`categories` の指定順ではなくアルファベット順になるので、  
必ず `target_names` でどちらが0か1かを確認する。

---

### `TfidfVectorizer(max_features=10000)`

TF-IDFスコアが高い上位N単語だけに絞る。  
全単語を使うとメモリが足りなくなるため。

---

### テストデータには `transform` だけ使う

```python
X_train = tfidf.fit_transform(X_train_raw)  # 学習データ：単語一覧を作って変換
X_test  = tfidf.transform(X_test_raw)       # テストデータ：変換だけ
```

テストデータで `fit` してしまうと「テストデータにしか出てこない単語」が  
単語一覧に入ってしまう。現実では未知データに学習時と同じ単語一覧で変換する必要があるため。

```
学習時：fit_transform → 単語一覧を作って変換
運用時：transform のみ → 学習時の単語一覧で変換
```

---

### `LogisticRegression(max_iter=1000)`

`max_iter` は最適化の繰り返し回数の上限。  
デフォルトの100だと収束しないことがあるので増やす。

---

## 今日の気づき

データ量が精度に直結する：
```
16件  → accuracy 40%（全部同じクラスと予測してしまう）
25000件 → accuracy 88%
```

---

## 疑問・メモ欄

### random_stateを指定しないと何が起きるか

```python
# random_stateなしで2回実行すると
# 1回目：たまたま簡単なテストデータ → 精度92%
# 2回目：たまたま難しいテストデータ → 精度84%
```

最悪のケース：何度も実行して精度が高く出た分割だけ報告する  
→「このモデルは精度92%」と言えてしまうが実際には再現しない

`random_state` の固定は「再現性の保証」という意味で研究・開発の基本マナー。

---

### `load_files` のラベル割り当てはアルファベット順

`categories=["pos", "neg"]` と指定しても `target_names` は `['neg', 'pos']`（アルファベット順）になる。  
`target_names` を確認しないと、最後に出力を見たとき0と1の定義を間違える可能性がある。

```python
print(train_data.target_names)
# → ['neg', 'pos']  ← neg=0, pos=1 と確認できる
```

**自分でラベルを指定したい場合**

方法1：ラベルを後から反転する
```python
y_train_flipped = 1 - y_train  # 0→1, 1→0 に反転
```

方法2：自分で読み込む
```python
import os

def load_reviews(base_path, label_map):
    texts, labels = [], []
    for label_name, label_value in label_map.items():
        folder = os.path.join(base_path, label_name)
        for filename in os.listdir(folder):
            with open(os.path.join(folder, filename), encoding="utf-8") as f:
                texts.append(f.read())
                labels.append(label_value)
    return texts, labels

# pos=0, neg=1 と自分で決める
texts, labels = load_reviews("aclImdb/train", {"pos": 0, "neg": 1})
```

ただし実務では `target_names` で確認する方が混乱が少ない。自分で指定すると後からコードを読んだ人が混乱することもある。