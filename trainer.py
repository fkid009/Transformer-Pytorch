import torch
from torch.nn import functional as F
import lightning as L

from data import PAD


class LitTransformer(L.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, src, tgt):
        return self.model(src, tgt)

    def compute_loss(self, batch):
        src, tgt = batch
        logits = self.model(src, tgt[:, :-1])  # (B, T, VOCAB_SIZE)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=PAD)

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
