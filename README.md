# NEAT SlimeVolley

NEAT アルゴリズムで SlimeVolley のエージェントを学習するプロジェクト。

## 必要条件

- Python 3.10–3.13（3.12 で開発・検証済み）
- 依存パッケージ

```bash
pip install -r requirements.txt
```

## ファイル構成

| ファイル | 説明 |
|---|---|
| `train_curriculum_parallel.py` | 並列評価で学習（メイン） |
| `train_curriculum_sequential.py` | 逐次評価で学習 |
| `play_vs_ai_network.py` | 学習済みネットワークの可視化（対戦・ネットワーク図表示） |
| `config-feedforward` | NEAT ハイパーパラメータ設定 |
| `requirements.txt` | 依存パッケージ一覧 |
| `ckpts_parallel/` | チェックポイント保存先（学習後に生成） |

## 使い方

```bash
# 訓練（並列、デフォルト）
python train_curriculum_parallel.py

# 訓練（逐次）
python train_curriculum_sequential.py

# AI vs 人間（GUI）
python play_vs_ai_network.py

# AI vs ベースラインAI（--auto）
python play_vs_ai_network.py --auto

# 特定のチェックポイントから可視化
python play_vs_ai_network.py --auto ./ckpts_parallel/ckpt-100
```

## 学習の仕組み

- **NEAT**: トポロジーも進化するニューラルネットワーク進化アルゴリズム
- **ラリー報酬**: ボールへの接近・返球で報酬、失点でペナルティ
- **カリキュラム**: ラリー継続を重視した報酬設計で段階的にスキル獲得
- **並列評価**: `ProcessPoolExecutor` で個体評価を並列化
