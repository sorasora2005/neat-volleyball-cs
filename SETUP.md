# 必要なソフトウェア

## Python
バージョン 3.10〜3.12 推奨（3.12 で開発・検証済み）

## 依存パッケージ
```bash
pip install -r requirements.txt
```

### 各パッケージの用途

| パッケージ | 使用するスクリプト | 用途 |
|---|---|---|
| neat-python | 全部 | NEAT アルゴリズム本体 |
| gym | 全部 | SlimeVolley 環境（slimevolleygym が要求） |
| gymnasium | 全部 | 警告抑制のための logger 設定のみ |
| slimevolleygym | 全部 | SlimeVolley 環境 |
| numpy | 全部 | 数値計算 |
| pygame | play_vs_ai_network.py | GUI 表示・キー入力 |
| opencv-python | play_vs_ai_network.py | ゲーム画面のリサイズ |
| Pillow | visualize_match.py | GIF 生成 |


## 起動方法
```bash
# 訓練（並列）
python experiments/0716/train_curriculum_parallel.py

# 訓練（逐次）
python experiments/0716/train_curriculum_sequential.py

# 可視化プレイ（AI vs 人間）
python experiments/0716/play_vs_ai_network.py

# 可視化プレイ（AI vs ベースライン、--auto）
python experiments/0716/play_vs_ai_network.py --auto

# チェックポイントを指定
python experiments/0716/play_vs_ai_network.py --auto ../ckpts_parallel/ckpt-793

# GIF 生成
python experiments/0716/visualize_match.py ../ckpts_parallel/ckpt-793 match.gif
```