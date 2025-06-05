# tiny_rnn_autoencoder.py
# -----------------------
# A minimal RNN autoencoder for 3-second motion snippets (shape = 180×2).
# Uses PyTorch. Designed for very small datasets (~36 snippets) as a proof of concept.
#
# Instructions:
# 1. Install PyTorch (CPU-only is fine) via:
#      pip install torch matplotlib numpy
# 2. Place your core_pool.npy (or any other <N,180,2> NumPy array) in the same folder.
# 3. Run:
#      python tiny_rnn_autoencoder.py
# 4. The script will train an encoder/decoder for 100 epochs and print train/val loss.
#    Then it will display a plot of an original vs. reconstructed snippet.

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F # Added for completeness, though not explicitly used by RNNAutoencoder directly
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────────────────────
# 1. Configuration
# ───────────────────────────────────────────────────────────────
np.random.seed(0)
torch.manual_seed(0)

DATA_PATH    = "core_pool.npy"   # Replace with your .npy file path
BATCH_SIZE   = 4
EPOCHS       = 100
LR           = 1e-3
CONFIG_HIDDEN_SIZE  = 16   # very small hidden dimension (Note: RNNAutoencoder below uses its own defaults)
CONFIG_LATENT_SIZE  = 8    # small bottleneck (Note: RNNAutoencoder below uses its own defaults)
CONFIG_NUM_LAYERS   = 1    # (Note: RNNAutoencoder below uses its own defaults)
SNIP_LEN     = 180  # frames per snippet (3 s @ 60 Hz)
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ───────────────────────────────────────────────────────────────
# 2. Dataset Definition
# ───────────────────────────────────────────────────────────────
class SnippetDataset(Dataset):
    def __init__(self, npy_path):
        data = np.load(npy_path)            # shape: (N, 180, 2)
        data = data.astype(np.float32)
        self.X = torch.from_numpy(data)     # shape: (N, 180, 2)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.X[idx]


# ───────────────────────────────────────────────────────────────
# 3. Model Definition
# ───────────────────────────────────────────────────────────────
# Note: These RNNEncoder and RNNDecoder are defined but not directly used by the RNNAutoencoder class below.
# The RNNAutoencoder class has its own internal GRU layers.
class RNNEncoder(nn.Module):
    def __init__(self, input_size=2, hidden_size=CONFIG_HIDDEN_SIZE, latent_size=CONFIG_LATENT_SIZE, num_layers=CONFIG_NUM_LAYERS):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        # x: (batch, seq_len=180, features=2)
        _, h_n = self.gru(x)           # h_n: (num_layers, batch, hidden_size)
        h_n = h_n[-1]                  # take last layer: (batch, hidden_size)
        z   = self.fc(h_n)             # (batch, latent_size)
        return z


class RNNDecoder(nn.Module):
    def __init__(self, latent_size=CONFIG_LATENT_SIZE, hidden_size=CONFIG_HIDDEN_SIZE, output_size=2, num_layers=CONFIG_NUM_LAYERS, seq_len=SNIP_LEN):
        super().__init__()
        self.seq_len   = seq_len
        self.fc_init   = nn.Linear(latent_size, hidden_size)   # initialize hidden state
        self.gru       = nn.GRU(input_size=output_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=True)
        self.output_fc = nn.Linear(hidden_size, output_size)

    def forward(self, z):
        # z: (batch, latent_size)
        batch_size = z.size(0)
        # initialize hidden state of GRU
        h_0 = torch.tanh(self.fc_init(z)).unsqueeze(0)  # (1, batch, hidden_size)
        # decoder input: zeros sequence (we rely on hidden state to generate)
        dec_input = torch.zeros(batch_size, self.seq_len, 2, device=z.device)  # (batch, 180, 2)
        # Run GRU
        out, _ = self.gru(dec_input, h_0)  # out: (batch, 180, hidden_size)
        # map to output dimension
        out = self.output_fc(out)          # (batch, 180, 2)
        return out

# This is the RNNAutoencoder class used in main()
class RNNAutoencoder(nn.Module):
    def __init__(self, hidden_size=32, latent_size=32, num_layers=2, teacher_forcing_ratio=0.8): # Default parameters
        super().__init__()

        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers  = num_layers
        self.tf_ratio    = teacher_forcing_ratio

        # (1) Optional: input normalization layer
        self.input_norm = nn.LayerNorm([SNIP_LEN, 2])  # normalizes each snippet per batch

        # (2) Encoder: 2‐layer GRU
        self.encoder_rnn = nn.GRU(
            input_size  = 2,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True
        )
        self.fc_enc = nn.Linear(hidden_size, latent_size)

        # (3) Decoder: 2‐layer GRU
        self.fc_dec = nn.Linear(latent_size, hidden_size * num_layers)
        self.decoder_rnn = nn.GRU(
            input_size  = 2,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True
        )
        self.fc_out = nn.Linear(hidden_size, 2)

    def forward(self, x):
        """
        x : (batch, SNIP_LEN, 2), float
        returns:
          recon : (batch, SNIP_LEN, 2)
          z     : (batch, latent_size)
        """
        batch = x.size(0)

        x_norm = self.input_norm(x)

        h0_enc = torch.zeros(self.num_layers, batch, self.hidden_size, device=x.device)
        _, h_enc = self.encoder_rnn(x_norm, h0_enc) # enc_out not used
        last_h = h_enc[-1, :, :]

        z = torch.tanh(self.fc_enc(last_h))

        dec_h_flat = self.fc_dec(z)
        dec_h = dec_h_flat.view(self.num_layers, batch, self.hidden_size)

        outputs = []
        y = torch.zeros(batch, 1, 2, device=x.device) # Start token

        for t in range(SNIP_LEN):
            out, dec_h = self.decoder_rnn(y, dec_h)
            y_pred = self.fc_out(out)

            use_tf = (torch.rand(1).item() < self.tf_ratio)
            if use_tf and self.training: # Teacher forcing only during training
                y = x[:, t].unsqueeze(1)
            else:
                y = y_pred
            outputs.append(y_pred)

        recon = torch.cat(outputs, dim=1)
        return recon, z


# ───────────────────────────────────────────────────────────────
# 4. Training Utility Functions
# ───────────────────────────────────────────────────────────────
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for xb, _ in dataloader: # Assuming (X, X) from SnippetDataset
        xb = xb.to(DEVICE)
        optimizer.zero_grad()
        x_hat, _ = model(xb) # We get recon (x_hat) and latent (z), ignore z for loss
        loss = criterion(x_hat, xb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    for xb, _ in dataloader: # Assuming (X, X)
        xb = xb.to(DEVICE)
        x_hat, _ = model(xb)
        loss = criterion(x_hat, xb)
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(dataloader.dataset)


# ───────────────────────────────────────────────────────────────
# 5. Main Training Loop
# ───────────────────────────────────────────────────────────────
def main():
    print(f"Using device: {DEVICE}")
    # 1) Load dataset
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please ensure core_pool.npy (or your specified .npy file) is in the same folder.")
        return

    dataset = SnippetDataset(DATA_PATH)
    N = len(dataset)
    if N < 4: # Need at least 2 for train, 1 for val, plus batch size considerations
        print(f"Error: Need at least 4 snippets for meaningful train/val split; found {N}")
        if N > 0 and N < BATCH_SIZE :
             print(f"Consider reducing BATCH_SIZE (currently {BATCH_SIZE}) if dataset is very small.")
        return

    # 2) Split into train/val (80/20)
    n_val   = max(1, N // 5)
    n_train = N - n_val
    
    if n_train == 0:
        print(f"Error: Not enough data for a training set after reserving {n_val} for validation. Total samples: {N}")
        return

    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)) # Added generator for reproducibility

    train_loader = DataLoader(train_ds, batch_size=min(BATCH_SIZE, n_train), shuffle=True) # ensure batch_size <= n_train
    val_loader   = DataLoader(val_ds,   batch_size=min(BATCH_SIZE, n_val), shuffle=False) # ensure batch_size <= n_val

    # 3) Instantiate model, optimizer, loss
    # This uses the RNNAutoencoder with its default parameters (hidden=32, latent=32, layers=2)
    model = RNNAutoencoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # 4) Training loop
    best_val_loss = float("inf")
    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss   = validate(model, val_loader, criterion)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_rnn_autoencoder.pth")
            print(f"Epoch {epoch:03d}: New best validation loss: {best_val_loss:.6f}. Model saved.")


    print(f"\nTraining complete. Best validation loss: {best_val_loss:.6f}")
    print(f"Best model saved to best_rnn_autoencoder.pth")

    # 5) Example: encode a snippet to its latent code (using the model state at end of training)
    print("\n--- Example: Encoding Snippet #0 (using final model state) ---")
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        sample_data, _ = dataset[0] # Get the first snippet from the original dataset
        sample_data_unsqueezed = sample_data.unsqueeze(0).to(DEVICE)  # Add batch dimension and send to device
        
        # Correctly call the model to get reconstruction and latent code
        _, z_latent = model(sample_data_unsqueezed) 
        
        print(f"Latent code of snippet #0 (shape: {z_latent.shape}):")
        print(z_latent.cpu().numpy())


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────
# (Post-Training) Plotting Section
# This section will run after main() completes.
# It loads the *best* saved model for plotting.
# ─────────────────────────────────────────────────────
print("\n--- Plotting Original vs. Reconstructed Snippet #0 (using best saved model) ---")

# Check if the model file and data file exist before trying to plot
if not os.path.exists("best_rnn_autoencoder.pth"):
    print("Error: 'best_rnn_autoencoder.pth' not found. Cannot run plotting section.")
    print("Please ensure training completed successfully and saved the model.")
elif not os.path.exists(DATA_PATH):
    print(f"Error: Data file '{DATA_PATH}' not found. Cannot run plotting section.")
else:
    # (A) Load the core pool snippets
    core_plot_data = np.load(DATA_PATH) # shape = (N, 180, 2)
    all_snips_plot = torch.from_numpy(core_plot_data).float() # Torch tensor (N, 180, 2)
    
    # We just need one snippet for plotting, e.g., the first one.
    if len(all_snips_plot) == 0:
        print("Error: No data in core_pool.npy to plot.")
    else:
        snippet_to_plot = all_snips_plot[0] # Get the first snippet (180, 2)

        # (B) Instantiate model and load weights
        # Ensure these parameters match the ones used during training for RNNAutoencoder
        plot_model = RNNAutoencoder().to(DEVICE) # Uses default (hidden_size=32, latent_size=32, num_layers=2)
        
        try:
            checkpoint = torch.load("best_rnn_autoencoder.pth", map_location=DEVICE)
            plot_model.load_state_dict(checkpoint)
            plot_model.eval()

            # (C) Helper to plot original vs. reconstructed (defined inline or could be top-level)
            def plot_reconstruction_trajectory(model_instance, single_snippet_tensor, device_to_use, snippet_id=0):
                model_instance.eval()
                with torch.no_grad():
                    # Add batch dimension and send to device
                    snippet_for_model = single_snippet_tensor.unsqueeze(0).to(device_to_use) # (1, 180, 2)
                    recon_output, _ = model_instance(snippet_for_model) # (1, 180, 2)
                    
                    recon_np = recon_output.cpu().numpy()[0] # (180, 2)
                    orig_np  = single_snippet_tensor.cpu().numpy() # (180, 2)

                plt.figure() # Create a new figure
                plt.plot(orig_np[:, 0], orig_np[:, 1], label="Original", alpha=0.75)
                plt.plot(recon_np[:, 0], recon_np[:, 1], "--", label="Reconstruction", alpha=0.75)
                plt.title(f"Original vs. Reconstructed Snippet #{snippet_id}")
                plt.xlabel("X Coordinate")
                plt.ylabel("Y Coordinate")
                plt.legend()
                plt.axis('equal') # Equal scaling for X and Y
                plt.grid(True)
                plt.show()

            # (D) Plot snippet #0
            plot_reconstruction_trajectory(plot_model, snippet_to_plot, DEVICE, snippet_id=0)
            print("Plotting complete. Check the displayed window.")

        except FileNotFoundError:
            print("Error: 'best_rnn_autoencoder.pth' could not be loaded for plotting. Was it saved correctly?")
        except Exception as e:
            print(f"An error occurred during plotting: {e}")