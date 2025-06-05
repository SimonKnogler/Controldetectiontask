This repository contains experiments and scripts for motion snippet generation.

## GAN Scripts

- `gan_motion_generator.py` – trains a simple unconditional GAN on `core_pool.npy`.
- `generate_gan_snippets.py` – loads a trained GAN and visualises generated snippets.
- `conditional_gan_motion_generator.py` – trains a conditional GAN using snippet
  labels from `core_pool_labels.npy`. The generator and discriminator both receive
  cluster labels so new snippets can be generated conditioned on a specific
  style cluster.
- `generate_cgan_snippets.py` – loads a conditional GAN generator and
  produces snippets for a chosen style label.

To run tests:

```bash
pytest -q
```
