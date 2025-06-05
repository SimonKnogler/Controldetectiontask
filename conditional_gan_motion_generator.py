# conditional_gan_motion_generator.py
# -----------------------------------
# Minimal conditional GAN example using PyTorch.
# Generates 3-second motion snippets (180x2) from core_pool.npy conditioned
# on cluster labels stored in core_pool_labels.npy.

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_PATH = os.path.join("Motion Library", "core_pool.npy")
LABEL_PATH = os.path.join("Motion Library", "core_pool_labels.npy")
BATCH_SIZE = 8
EPOCHS = 500
LATENT_DIM = 32
HIDDEN_SIZE = 64
NUM_CLASSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LabeledDataset(Dataset):
    """Dataset returning (snippet, label)."""
    def __init__(self, data_path: str, label_path: str):
        data = np.load(data_path).astype(np.float32)
        labels = np.load(label_path).astype(np.int64)
        if len(labels) < len(data):
            labels = labels[: len(data)]
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Generator(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, hidden_size: int = HIDDEN_SIZE, seq_len: int = 180):
        super().__init__()
        self.embed = nn.Embedding(NUM_CLASSES, hidden_size)
        self.fc = nn.Linear(latent_dim + hidden_size, hidden_size)
        self.rnn = nn.GRU(2, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, 2)
        self.seq_len = seq_len

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cond = self.embed(labels)
        h0 = torch.tanh(self.fc(torch.cat([z, cond], dim=1))).unsqueeze(0)
        x = torch.zeros(z.size(0), self.seq_len, 2, device=z.device)
        out, _ = self.rnn(x, h0)
        return self.out(out)


class Discriminator(nn.Module):
    def __init__(self, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.embed = nn.Embedding(NUM_CLASSES, hidden_size)
        self.rnn = nn.GRU(2, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cond = self.embed(labels)
        _, h = self.rnn(x)
        h = h[-1] + cond
        return self.fc(h)


def train() -> None:
    if not os.path.exists(DATA_PATH) or not os.path.exists(LABEL_PATH):
        print("Data or label file missing")
        return

    ds = LabeledDataset(DATA_PATH, LABEL_PATH)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4)
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4)
    criterion = nn.BCEWithLogitsLoss()

    real_label = 1.
    fake_label = 0.

    for epoch in range(1, EPOCHS + 1):
        for real_snip, lbl in dl:
            real_snip = real_snip.to(DEVICE)
            lbl = lbl.to(DEVICE)

            # Train Discriminator
            opt_D.zero_grad()
            out_real = D(real_snip, lbl)
            loss_real = criterion(out_real.squeeze(), torch.full((real_snip.size(0),), real_label, device=DEVICE))

            z = torch.randn(real_snip.size(0), LATENT_DIM, device=DEVICE)
            fake_snip = G(z, lbl)
            out_fake = D(fake_snip.detach(), lbl)
            loss_fake = criterion(out_fake.squeeze(), torch.full((real_snip.size(0),), fake_label, device=DEVICE))
            loss_D = loss_real + loss_fake
            loss_D.backward()
            opt_D.step()

            # Train Generator
            opt_G.zero_grad()
            out_fake = D(fake_snip, lbl)
            loss_G = criterion(out_fake.squeeze(), torch.full((real_snip.size(0),), real_label, device=DEVICE))
            loss_G.backward()
            opt_G.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{EPOCHS} | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")

    torch.save(G.state_dict(), "cgan_generator.pth")
    print("Training complete. Generator saved to cgan_generator.pth")


if __name__ == "__main__":
    train()
