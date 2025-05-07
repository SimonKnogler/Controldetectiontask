import numpy as np
from pathlib import Path

# determine script’s own folder
script_dir = Path(__file__).parent

# look for motion_donors next to the script
data_dir   = script_dir / "motion_donors"
files      = sorted(data_dir.glob("*.npy"))

if not files:
    raise RuntimeError(f"No files found in {data_dir}")

# load & concatenate
all_snips = np.concatenate([np.load(f) for f in files], axis=0)

# save master library
out_path = script_dir / "master_snippets.npy"
np.save(out_path, all_snips.astype(np.float16))
print("Master library shape:", all_snips.shape)
print("Saved to", out_path)




import numpy as np
import matplotlib.pyplot as plt

# 1) Load the master library
snips = np.load("master_snippets.npy")  # shape (9,300,2)

# 2) Pick one snippet to inspect (e.g. the first)
snippet = snips[3]  # shape (300,2)

# 3) Plot the XY trajectory
plt.figure(figsize=(5,5))
plt.plot(snippet[:,0], snippet[:,1], '-o', markersize=2)
plt.title("Snippet 0: X vs Y Trajectory")
plt.xlabel("X position (px)")
plt.ylabel("Y position (px)")
plt.axis('equal')
plt.grid(True)
plt.show()

# 4) Plot X and Y over time
frames = np.arange(snippet.shape[0])
plt.figure(figsize=(8,3))
plt.plot(frames, snippet[:,0], label="X")
plt.plot(frames, snippet[:,1], label="Y")
plt.title("Snippet 0: X & Y over Time")
plt.xlabel("Frame")
plt.ylabel("Position (px)")
plt.legend()
plt.tight_layout()
plt.show()





