# NLP自己学習ロードマップ

## 概要

機械学習・自然言語処理の基礎から、翻訳・テキスト生成まで到達することを目標とした17日間の自己学習記録。

- 学習時間：1日あたり約2時間
- 環境：Python 3.12 / PyTorch / HuggingFace Transformers
- 最終目標：翻訳・テキスト生成パイプラインの実装

---

## 進捗

| Day | テーマ | 状況 |
|-----|--------|------|
| 1 | ML基礎・線形回帰 | ✅ 完了 |
| 2 | テキスト前処理・BoW・TF-IDF | ✅ 完了 |
| 3 | 形態素解析・日本語NLP | ✅ 完了 |
| 4 | ロジスティック回帰・テキスト分類・評価指標 | ✅ 完了 |
| 5 | 単語埋め込み・Word2Vec・コサイン類似度 | ✅ 完了 |
| 6 | ニューラルネット基礎・順伝播・逆伝播 | ✅ 完了 |
| 7 | PyTorch基本操作・nn.Module・optimizer | ✅ 完了 |
| 8 | RNN・LSTM・勾配消失問題 | ✅ 完了 |
| 9 | Attentionメカニズム・Q・K・V | ✅ 完了 |
| 10 | Transformerの全体構造 | ✅ 完了 |
| 11 | BERT入門・HuggingFaceの使い方 | ✅ 完了 |
| 12 | 日本語BERTでテキスト分類 | ✅ 完了 |
| 13 | GPTの構造・テキスト生成・生成パラメータ | ✅ 完了 |
| 14 | 翻訳モデルの実践（Helsinki-NLP） | 🔜 次 |
| 15 | テキスト生成の実践・生成パラメータ調整 | ⬜ 未着手 |
| 16 | ミニプロジェクト：翻訳→要約パイプライン | ⬜ 未着手 |
| 17 | 振り返り・次のステップの展望 | ⬜ 未着手 |

---

## 各Dayのまとめ

### Day 1：ML基礎・線形回帰
線形回帰の仕組みを最小二乗法から理解。scikit-learnの `fit` / `predict` が何をしているかを手で追った。標準化・多重共線性の概念も学習。

**キーワード**：最小二乗法・重回帰・標準化・多重共線性・reshape・fit・predict

---

### Day 2：テキスト前処理・BoW・TF-IDF
テキストを数値に変換する方法を学習。BoWは単純な出現回数カウント、TF-IDFは「その文書らしい単語」を浮き上がらせる手法。日本語はスペースがないので形態素解析が必要と気づく。

**キーワード**：Bag of Words・TF-IDF・CountVectorizer・TfidfVectorizer・疎行列

---

### Day 3：形態素解析・日本語NLP
janomeで日本語の形態素解析を実装。品詞フィルタリング・原形への統一・TF-IDFとの組み合わせを実践。

**キーワード**：形態素・形態素解析・janome・品詞フィルタリング・原形・WordPiece

---

### Day 4：ロジスティック回帰・テキスト分類
シグモイド関数でロジスティック回帰の仕組みを理解。IMDb 25,000件のレビューデータで感情分析を実装し精度88%を達成。評価指標（accuracy・precision・recall・F1）の読み方を学習。

**キーワード**：シグモイド関数・ロジスティック回帰・train_test_split・BCELoss・precision・recall・F1

---

### Day 5：単語埋め込み・Word2Vec
「同じ文脈に出てくる単語は意味が近い」という仕組みでWord2Vecを理解。コサイン類似度で単語の近さを計算し、`king - man + woman = queen` のベクトル演算を体感。PCAで300次元を2次元に圧縮して可視化。

**キーワード**：単語埋め込み・Word2Vec・コサイン類似度・PCA・most_similar

---

### Day 6：ニューラルネット基礎
線形変換だけでは表現力が増えない → 活性化関数が必要という理解から出発。ReLU・シグモイドの特徴を比較。NumPyで順伝播を手実装し、PyTorchの自動微分で逆伝播を体験。損失が0.0215 → 0.0068に下がることを確認。

**キーワード**：活性化関数・ReLU・シグモイド・順伝播・逆伝播・勾配降下法・自動微分・テンソル

---

### Day 7：PyTorch基本操作・nn.Module
Day 6の手動実装をnn.Moduleで整理。optimizerで重みの更新を自動化。`zero_grad` → `backward` → `step` の3行で完結するようになった。

**キーワード**：nn.Module・nn.Linear・optimizer・SGD・BCELoss・forward・__init__・super()

---

### Day 8：RNN・LSTM・勾配消失問題
RNNで可変長テキストを処理する仕組みを理解。BPTTで長い文章では勾配消失が起きる原因を学習。LSTMのゲート機構（忘却・入力・出力）で勾配消失を緩和する仕組みを理解。

**キーワード**：RNN・LSTM・BPTT・勾配消失・勾配爆発・cell状態・hidden状態・ゲート機構

---

### Day 9：Attentionメカニズム
「図書館の検索」の例でQ・K・Vの役割を直感的に理解。全単語に直接アクセスできるAttentionはRNNの距離問題を根本から解決する。softmaxでスコアを確率に変換して各単語の情報を重み付きで混ぜ合わせる仕組みを手実装。

**キーワード**：Attention・Query・Key・Value・softmax・コサイン類似度・Self-Attention

---

### Day 10：Transformerの全体構造
位置エンコーディング（sin/cos）でAttentionに順番情報を付加。Multi-Head Attentionで複数の視点から文脈を捉える仕組みを実装。Add & Norm（残差接続）・Feed Forward Networkの役割を理解。BERTはエンコーダのみ・GPTはデコーダのみという構造を確認。

**キーワード**：位置エンコーディング・Multi-Head Attention・残差接続・FFN・エンコーダ・デコーダ

---

### Day 11：BERT入門・HuggingFace
事前学習・ファインチューニングの概念を理解。HuggingFaceのpipelineで5行で感情分析を実装。WordPiece（頻出単語はそのまま・珍しい単語はサブワードに分割）の仕組みを確認。pipelineの内部処理（tokenizer→model→softmax→ラベル変換）を自分で再現。

**キーワード**：事前学習・ファインチューニング・HuggingFace・pipeline・WordPiece・[CLS]・[SEP]・logits

---

### Day 12：日本語BERTでテキスト分類
日本語BERT（3クラス：POSITIVE/NEGATIVE/NEUTRAL）で感情分析を実装。WordPieceの「基本形はそのまま・活用形は分割」パターンが日英共通であることを発見。文末バイアスの原因が学習データの偏りにあることを理解。

**キーワード**：日本語BERT・3クラス分類・WordPieceの日英共通パターン・文末バイアス・ドメイン特化モデル

---

### Day 13：GPTの構造・テキスト生成
BERTとGPTの構造的な違いを理解。Causal Maskで未来のトークンをマスクして左から右に生成する仕組みを確認。temperature・top_k・top_pはすべて「候補を絞って確率に従って1つ選ぶ」ための手法。

**キーワード**：GPT・Causal Mask・temperature・top_k・top_p・do_sample・greedy decoding

---

## 環境

```
OS      ：Ubuntu (WSL2)
Python  ：3.12.3
主要ライブラリ：
  torch
  transformers
  scikit-learn
  numpy / pandas
  matplotlib
  gensim
  janome
```

---

## リポジトリ構成

```
.
├── README.md
├── day01/
│   ├── day01.ipynb
│   └── day01.md
├── day02/
│   ├── day02.ipynb
│   └── day02.md
...
├── day13/
│   ├── day13.ipynb
│   └── day13.md
└── .gitignore
```