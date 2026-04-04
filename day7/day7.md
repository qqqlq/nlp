# Day 7：PyTorchの基本操作・nn.Module

## 今日やったこと
- `nn.Module` でネットワークをクラスとして定義
- `optimizer` で重み更新を自動化
- Day 6の手動実装との対応関係を理解

---

## Day 6 vs Day 7の比較

```python
# Day 6：手動で全部書く
W1.grad = None
b1.grad = None
W2.grad = None
b2.grad = None
loss.backward()
with torch.no_grad():
    W1 -= lr * W1.grad
    b1 -= lr * b1.grad
    W2 -= lr * W2.grad
    b2 -= lr * b2.grad

# Day 7：3行で完結
optimizer.zero_grad()  # 勾配リセット
loss.backward()        # 逆伝播
optimizer.step()       # 重み更新
```

やっていることの本質は同じ。`optimizer` が全パラメータを自動管理しているので層が増えても3行のまま。

---

## nn.Moduleでネットワークを定義する

```python
class SimpleNet(nn.Module):        # nn.Moduleを継承
    def __init__(self):            # インスタンス化時に自動実行
        super().__init__()         # 親クラス（nn.Module）の初期化
        self.layer1 = nn.Linear(2, 3)  # 入力2 → 隠れ層3
        self.layer2 = nn.Linear(3, 1)  # 隠れ層3 → 出力1

    def forward(self, x):          # 順伝播の計算を定義
        h = torch.relu(self.layer1(x))
        y = torch.sigmoid(self.layer2(h))
        return y

model = SimpleNet()  # インスタンス化
```

---

## キーワード集

### `__init__(self)`
インスタンス化時に自動で呼ばれる初期化処理。

```python
model = SimpleNet()  # この瞬間に __init__() が自動で呼ばれる
```

### `super().__init__()`
親クラス（nn.Module）の初期化処理を実行する。これを書かないとnn.Moduleの内部機能（パラメータ管理など）が使えなくなる。

### `self`
「このインスタンス自身」を指す。

```python
self.layer1 = nn.Linear(2, 3)
# model.layer1 と self.layer1 は同じものを指している
```

### `forward(x)`
順伝播の計算を定義するメソッド。直接呼ばずに `model(x)` と書くのが一般的。

```python
model.forward(x)  # 直接呼ぶ（あまり使わない）
model(x)          # こっちが一般的（内部でforward()を呼ぶ）
```

`model(x)` の方がフック（前後に処理を挟む仕組み）なども実行してくれる。

### `nn.Linear(in, out)`
線形変換の層。重みとバイアスを自動で管理する。

```python
self.layer1 = nn.Linear(2, 3)  # 入力2 → 出力3
# 内部で W(2×3) と b(3,) を持っている
```

### `model.parameters()`
モデルの全パラメータ（重み・バイアス）をまとめて返す。optimizerに渡すことで全パラメータを自動管理できる。

### `torch.optim.SGD`
確率的勾配降下法（SGD）のoptimizer。

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

### `optimizer.zero_grad()`
全パラメータの勾配をリセットする。Day 6の `W1.grad = None` に相当。

### `optimizer.step()`
全パラメータを勾配に従って更新する。Day 6の `W1 -= lr * W1.grad` に相当。

### `nn.BCELoss()`
二値分類（Binary Cross Entropy）の損失関数。

```python
loss_fn = nn.BCELoss()
loss = loss_fn(y_pred, y_true)
```

### `grad_fn`
テンソルに表示される計算過程の記録。

```python
tensor([0.4944], grad_fn=<SigmoidBackward0>)
#                ↑ Sigmoidの計算結果であることが記録されている
```

---

## 学習ループの標準的な書き方

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

for i in range(100):
    # 1. 順伝播
    y_pred = model(x)
    loss = loss_fn(y_pred, y_true)

    # 2. 勾配リセット → 逆伝播 → パラメータ更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## Day 6・Day 7の対応関係まとめ

| Day 6 | Day 7 |
|-------|-------|
| `W1.grad = None` など手動で全変数 | `optimizer.zero_grad()` |
| `loss.backward()` | `loss.backward()` （同じ） |
| `with torch.no_grad(): W1 -= lr * W1.grad` など | `optimizer.step()` |
| `W1 = torch.randn(..., requires_grad=True)` など | `nn.Linear` が自動管理 |

---

## 疑問・メモ欄

### 損失関数の使い分け

| 用途 | 損失関数 |
|------|---------|
| 回帰（数値予測） | MSE（二乗誤差） |
| 二値分類 | BCELoss（二値交差エントロピー） |
| 多クラス分類 | CrossEntropyLoss |

**なぜ二値分類にMSEを使わないか**

シグモイドは両端で飽和する（勾配がほぼ0になる）ため、予測が大きく外れているときにMSEだと学習がほぼ止まってしまう。

BCELossは `log` の性質により外れているほど損失が急激に大きくなるので強く修正される → 学習が速く進む。

```
y_true=1 のとき
  y_pred=0.99 → loss ≈ 0.01  （正解に近い → 損失小）
  y_pred=0.50 → loss ≈ 0.69  （曖昧 → 損失中）
  y_pred=0.01 → loss ≈ 4.60  （大きく外れ → 損失大）
```

**損失関数の引数の順番に注意**

MSEは順番が関係ないが、BCELossは非対称なので順番が重要。

```python
loss_fn(y_pred, y_true)  # ✅ 予測値が先
loss_fn(y_true, y_pred)  # ❌ 全く別の値になる
```

損失関数によってルールが違うのでドキュメントで確認する癖をつける。

---

### 膨大な層のネットワークはどう書くのか？

```python
# nn.Sequentialで連続した層をまとめられる
self.network = nn.Sequential(
    nn.Linear(2, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)

# BERTなど巨大なモデルはHuggingFaceから一行でロードする
from transformers import BertModel
model = BertModel.from_pretrained("bert-base-uncased")
# → 12層・1億パラメータのネットワークが一行で使える
```

詳細は **Day 11（BERT・HuggingFace）** で。