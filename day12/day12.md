# Day 12：日本語BERTでテキスト分類

## 今日やったこと
- 日本語BERTで感情分析を実装
- 日本語トークナイザーの挙動を確認
- WordPieceの日英共通パターンを発見
- 複数レビューを分類して集計

---

## 英語BERTと日本語BERTの違い

```
英語BERT：英語テキストで事前学習 → 英語のみ対応
          WordPiece（スペース区切りベース）

日本語BERT：日本語テキストで事前学習 → 日本語に対応
            文字単位 or 形態素解析ベース
            → スペースがないので形態素解析が必要
```

---

## 使ったモデル

```
koheiduck/bert-japanese-finetuned-sentiment

日本語BERT → 日本語テキストで事前学習
3クラス分類：POSITIVE / NEGATIVE / NEUTRAL
```

---

## 感情分析の実装

```python
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="koheiduck/bert-japanese-finetuned-sentiment"
)

texts = [
    "この映画はとても面白かった！",
    "最悪な映画だった。時間の無駄。",
    "まあまあかな。普通だった。",
]

results = classifier(texts)
for text, result in zip(texts, results):
    print(f"{result['label']} ({result['score']:.3f}): {text}")
```

出力：
```
POSITIVE (0.993): この映画はとても面白かった！
NEGATIVE (0.995): 最悪な映画だった。時間の無駄。
NEUTRAL  (0.944): まあまあかな。普通だった。
```

---

## 日本語トークナイザーの挙動

```python
tokenizer.tokenize("この映画はとても面白かった！")
# → ['この', '映画', 'は', 'とても', '面白', '##かっ', 'た', '!']

tokenizer.tokenize("感動して泣いてしまいました。")
# → ['感動', 'し', 'て', '泣い', 'て', 'しまい', 'まし', 'た', '。']
```

### 英語との共通パターン（WordPiece）

基本形（辞書形）は頻出なのでそのまま、活用形は比較的珍しいので分割される。

```
すごい   → ['すごい']      ← 形容詞の基本形は頻出
すごく   → ['すご', '##く'] ← 副詞形は少し珍しい

面白い   → ['面白い']          ← 基本形は頻出
面白かった → ['面白', '##かっ', 'た'] ← 過去形は活用形なので分割
面白く   → ['面白', '##く']    ← 副詞形は分割
```

英語の例と同じパターン：
```
unbelievable → そのまま（形容詞の基本形）
unbelievably → 分割（副詞形）
```

---

## スコアの読み方

スコアが低いときは「迷いがある」ことを示す。

```
NEGATIVE (0.995): 最悪な映画だった。    ← 明確にネガティブ
NEGATIVE (0.778): 普通かな。印象に残らなかった。← 迷いがある

「印象に残らなかった」という否定表現でNEGATIVEに引っ張られているが
完全にネガティブとは言い切れないのでスコアが低め
```

似ている文章でも判定が変わる例：
```
「まあまあかな。普通だった。」       → NEUTRAL (0.944)  中立的なニュアンス
「普通かな。特に印象に残らなかった。」 → NEGATIVE (0.778) 否定表現が含まれる
```

---

## 集計の実装

```python
from collections import Counter

results = classifier(reviews)
labels = [r['label'] for r in results]
counts = Counter(labels)
print(dict(counts))
# → {'POSITIVE': 4, 'NEGATIVE': 4}
```

### `Counter` とは
リストの要素の出現回数を数えて辞書形式で返す。

```python
Counter(['A', 'B', 'A', 'C', 'A'])
# → Counter({'A': 3, 'B': 1, 'C': 1})
```

---

## 今日の流れまとめ

```
1. 日本語BERTは日本語テキストで事前学習・形態素解析ベースのトークナイザー
2. WordPieceの「基本形はそのまま・活用形は分割」パターンは日英共通
3. 3クラス分類（POSITIVE/NEGATIVE/NEUTRAL）で感情分析
4. スコアが低いほどモデルが迷っている → 境界的な文章
5. Counterで分類結果を集計できる
```

---

## 疑問・メモ欄

### なぜBERTは文末の表現に引っ張られやすいのか

理論上はAttentionで全単語を平等に見られるので構造的な問題ではない。

原因は**学習データの偏り**。

```
学習データの統計的パターン：
「まあまあかな」「良かった」などの評価表現が文末に来ることが多い
→ モデルが「文末の評価表現 = 全体の感情」と学習してしまう
```

理論（Attention）と実際の挙動（文末バイアス）のギャップは
学習データの偏りから来ることが多い。

アンビバレントな文章（良い面と悪い面が混在）への対処法：
```
① より多くのデータでファインチューニング
② アスペクトベース感情分析（「アクション」「内容」を別々に評価）
``` 