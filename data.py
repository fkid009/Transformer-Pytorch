import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

PAD, SOS, EOS = 0, 1, 2


class CharTokenizer:
    def __init__(self, texts):
        self.chars = sorted(set("".join(texts)))
        self.vocab_size = len(self.chars) + 3  # PAD, SOS, EOS
        self.stoi = {ch: i + 3 for i, ch in enumerate(self.chars)}  # string to int
        self.itos = {i: ch for ch, i in self.stoi.items()}  # int to string

    def encode(self, s):
        return [SOS] + [self.stoi[c] for c in s] + [EOS]

    def decode(self, ids):
        return "".join(self.itos.get(i, "") for i in ids)  # special token은 버림


def load_data():
    return load_dataset("bentrevett/multi30k")


def collate(batch):
    # 배치 안에서 가장 긴 문장 길이에 맞춰 PAD
    src, tgt = zip(*batch)
    return (
        pad_sequence(src, batch_first=True, padding_value=PAD),
        pad_sequence(tgt, batch_first=True, padding_value=PAD),
    )


def build_dataloaders(dataset, tokenizer, max_len, batch_size, num_workers=2):
    # 처음에 한 번만 인코딩. (src, tgt) 텐서 쌍의 list는 그대로 Dataset으로 쓸 수 있음
    def encode(split):
        return [
            (torch.tensor(tokenizer.encode(x["en"])[:max_len]), torch.tensor(tokenizer.encode(x["de"])[:max_len]))
            for x in dataset[split]
        ]

    train_loader = DataLoader(encode("train"), batch_size=batch_size, shuffle=True, drop_last=True,
                              collate_fn=collate, num_workers=num_workers)
    val_loader = DataLoader(encode("validation"), batch_size=batch_size, shuffle=False,
                            collate_fn=collate, num_workers=num_workers)
    return train_loader, val_loader
