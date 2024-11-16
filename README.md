# myvae

VAEの学習コードです。

## Usage

```bash
$ git clone https://github.com/hnmr293/myvae.git
# `uv` を使う場合
$ uv sync --frozen
# `pip` を使う場合
$ python -m venv .venv --prompt "myvae"
$ . .venv/bin/activate
(myvae) $ pip install -r requirements.txt
```

## Train

事前に `wandb` にログインしておきます。自動で投稿したくない場合はオフラインモードにしておきます。

```bash
$ wandb login
# 任意
$ wandb offline
```

`config/train.toml` を編集し、 `accelerate launch` で `main.py` を実行します。

```bash
# `uv` を使う場合
$ uv run accelerate launch main.py config/train.yaml
# `pip` を使う場合
(myvae) $ accelerate launch main.py config/train.yaml
```
