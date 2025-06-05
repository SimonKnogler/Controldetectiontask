
#------Plotting example trajectories----------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# 1) Load the core pool
core = np.load("core_pool.npy")       # shape (N, SNIP_LEN, 2)
print("Loaded core pool:", core.shape)

# 2) Plot a few example trajectories (X vs. Y)
for i in [0, 1, 2]:                   # first three snippets
    traj = core[i]                    # shape (SNIP_LEN, 2)
    plt.figure()
    plt.plot(traj[:,0], traj[:,1], lw=1)
    plt.title(f"Snippet #{i} Trajectory")
    plt.xlabel("X (z-scored units)")
    plt.ylabel("Y (z-scored units)")
    plt.gca().set_aspect('equal', 'box')
plt.show()



# ─── Load the core pool ─────────────────────────────────
# Adjust the path if core_pool.npy is in a different folder
core = np.load("core_pool.npy")   # shape (N, SNIP_LEN, 2)
traj = core[0]                    # pick the first snippet

# ─── Set up the figure ─────────────────────────────────
fig, ax = plt.subplots()
ln, = ax.plot([], [], 'o-', lw=2)
ax.set_xlim(traj[:, 0].min() * 1.1, traj[:, 0].max() * 1.1)
ax.set_ylim(traj[:, 1].min() * 1.1, traj[:, 1].max() * 1.1)
ax.set_aspect('equal', 'box')
ax.set_title("3-Second Snippet Animation")
ax.set_xlabel("X (z-scored)")
ax.set_ylabel("Y (z-scored)")

# ─── Initialization function ───────────────────────────
def init():
    ln.set_data([], [])
    return ln,

# ─── Update function ───────────────────────────────────
def update(frame):
    # Plot trajectory up to current frame
    ln.set_data(traj[:frame, 0], traj[:frame, 1])
    return ln,

# ─── Compute interval so total animation lasts 3 seconds ─
total_frames = traj.shape[0]            # e.g. 180 frames
interval_ms  = 6000 / total_frames      # ms per frame

# ─── Create the animation ──────────────────────────────
ani = animation.FuncAnimation(
    fig,
    update,
    frames=range(1, total_frames + 1),
    init_func=init,
    blit=True,
    interval=interval_ms
)

plt.show()