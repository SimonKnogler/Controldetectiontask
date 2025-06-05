# gan_motion_generator.py
# -----------------------
# Simple GAN to generate 3-second motion snippets (180x2) from core_pool.npy.
# Uses PyTorch. This is a minimal example for small datasets.

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Configuration
DATA_PATH = os.path.join("Motion Library", "core_pool.npy")
BATCH_SIZE = 8
EPOCHS = 2000
LATENT_DIM = 32
HIDDEN_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SnippetDataset(Dataset):
    """Dataset returning (180,2) float32 snippets."""
    def __init__(self, npy_path):
        arr = np.load(npy_path).astype(np.float32)
        self.data = torch.from_numpy(arr)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, hidden_size=HIDDEN_SIZE, seq_len=180):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_dim, hidden_size)
        self.rnn = nn.GRU(2, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, 2)

    def forward(self, z):
        batch = z.size(0)
        h0 = torch.tanh(self.fc(z)).unsqueeze(0)
        x = torch.zeros(batch, self.seq_len, 2, device=z.device)
        out, _ = self.rnn(x, h0)
        return self.out(out)


class Discriminator(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.rnn = nn.GRU(2, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, h = self.rnn(x)
        h = h[-1]
        return self.fc(h)


def train():
    if not os.path.exists(DATA_PATH):
        print(f"Data file {DATA_PATH} not found")
        return

    ds = SnippetDataset(DATA_PATH)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4)
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4)
    criterion = nn.BCEWithLogitsLoss()

    real_label = 1.
    fake_label = 0.

    for epoch in range(1, EPOCHS + 1):
        for real_snip in dl:
            real_snip = real_snip.to(DEVICE)

            # Train Discriminator
            opt_D.zero_grad()
            out_real = D(real_snip)
            loss_real = criterion(out_real.squeeze(), torch.full((real_snip.size(0),), real_label, device=DEVICE))

            z = torch.randn(real_snip.size(0), LATENT_DIM, device=DEVICE)
            fake_snip = G(z)
            out_fake = D(fake_snip.detach())
            loss_fake = criterion(out_fake.squeeze(), torch.full((real_snip.size(0),), fake_label, device=DEVICE))
            loss_D = loss_real + loss_fake
            loss_D.backward()
            opt_D.step()

            # Train Generator
            opt_G.zero_grad()
            out_fake = D(fake_snip)
            loss_G = criterion(out_fake.squeeze(), torch.full((real_snip.size(0),), real_label, device=DEVICE))
            loss_G.backward()
            opt_G.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{EPOCHS} | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")

    torch.save(G.state_dict(), "gan_generator.pth")
    print("Training complete. Generator saved to gan_generator.pth")


if __name__ == "__main__":
    train()
