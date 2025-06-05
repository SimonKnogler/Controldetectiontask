import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

LATENT_DIM = 32
HIDDEN_SIZE = 64
SEQ_LEN = 180
NUM_CLASSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, hidden_size: int = HIDDEN_SIZE, seq_len: int = SEQ_LEN):
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

def generate_cgan_snippets(generator_path: str = "cgan_generator.pth", label: int = 0, num_samples: int = 1) -> np.ndarray:
    """Load the conditional generator and sample new snippets for a given label."""
    generator = Generator(latent_dim=LATENT_DIM, hidden_size=HIDDEN_SIZE, seq_len=SEQ_LEN)
    generator.load_state_dict(torch.load(generator_path, map_location=DEVICE))
    generator.to(DEVICE)
    generator.eval()

    labels = torch.full((num_samples,), label, dtype=torch.long, device=DEVICE)
    z_noise = torch.randn(num_samples, LATENT_DIM, device=DEVICE)
    with torch.no_grad():
        generated = generator(z_noise, labels).cpu().numpy()
    print(f"Generated {generated.shape[0]} snippets for label {label}.")

    for i in range(num_samples):
        plt.figure(figsize=(6, 4))
        plt.plot(generated[i, :, 0], generated[i, :, 1])
        plt.title(f"Label {label} Snippet #{i+1}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.axis("equal")
        plt.grid(True)
        plt.show()

    return generated

if __name__ == "__main__":
    try:
        print("Attempting to generate snippets from conditional model...")
        generate_cgan_snippets(num_samples=3, label=0)
    except FileNotFoundError:
        print("Generator model file 'cgan_generator.pth' not found. Please train the model first by running conditional_gan_motion_generator.py.")
    except Exception as exc:
        print(f"An error occurred during generation: {exc}")

