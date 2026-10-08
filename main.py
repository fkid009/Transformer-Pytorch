import time

import torch
import lightning as L

from data import PAD, CharTokenizer, load_data, build_dataloaders
from transformer import Transformer
from trainer import LitTransformer

MAX_LEN = 256       # 글자 수 기준 (SOS/EOS 포함)
BATCH_SIZE = 64

D_MODEL = 256
N_HEADS = 8
N_LAYERS = 3
DROPOUT = 0.1

LR = 3e-4
MAX_EPOCHS = 10


def main():
    L.seed_everything(42)

    dataset = load_data()
    texts = [x[lang] for split in ("train", "validation") for x in dataset[split] for lang in ("en", "de")]
    tokenizer = CharTokenizer(texts)  # en/de 공유 vocab
    train_loader, val_loader = build_dataloaders(dataset, tokenizer, MAX_LEN, BATCH_SIZE)

    model = Transformer(tokenizer.vocab_size, tokenizer.vocab_size, D_MODEL, N_HEADS, N_LAYERS, MAX_LEN, PAD, DROPOUT)
    lit_model = LitTransformer(model, LR)

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=MAX_EPOCHS,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        gradient_clip_val=1.0,    # gradient 폭주 방지
        log_every_n_steps=50,
    )

    start = time.time()
    trainer.fit(lit_model, train_loader, val_loader)
    print(f"학습 시간: {(time.time() - start) / 60:.1f}분")
    if torch.cuda.is_available():
        print(f"최대 GPU 메모리: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__": 
    main()
