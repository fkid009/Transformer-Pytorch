# Transformer from Scratch

A minimal PyTorch implementation of the Transformer architecture from [Attention Is All You Need](https://arxiv.org/abs/1706.03762), trained on English-German translation (Multi30k).

## Project Structure

```
attention.py      Multi-Head Attention
embedding.py      Token Embedding + Positional Encoding
encoder.py        Encoder Layer & Encoder
decoder.py        Decoder Layer & Decoder
transformer.py    Full Transformer model
data.py           Char tokenizer & dataloaders
trainer.py        Lightning module
main.py           Training entry point
```

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## Hyperparameters

| Parameter | Default |
|-----------|---------|
| d_model | 256 |
| n_heads | 8 |
| n_layers | 3 |
| max_len | 256 (chars) |
| batch_size | 64 |
| lr | 3e-4 |
| max_epochs | 10 |

Edit `main.py` to change hyperparameters.
