# Generative Models — Diffusion & VAE

Two notebooks exploring image generation from scratch in PyTorch.

## Notebooks

### `Gen_with_diffusion.ipynb` — Conditional Diffusion on ASL Alphabet
A class-conditional **DDPM** (denoising diffusion model) that generates American Sign
Language hand-sign images, conditioned on the letter you ask for.

- **Dataset:** ASL Alphabet (29 classes: A–Z + `del`, `nothing`, `space`)
- **Backbone:** U-Net with time + label embeddings injected at the bottleneck
- **Image size:** 32×32 RGB
- **Schedule:** linear β, 1000 noise steps
- Trained for 150 epochs with checkpointing + resume support

### `Gen_with_VAE.ipynb` — VAE on MNIST
A variational autoencoder trained on MNIST for handwritten-digit generation.
*(Edit this line to match what the notebook actually does.)*

## Pretrained Weights

The trained `.pth` files are too large for GitHub, so they live on Google Drive:

**➡️ [Download weights (.pth)](https://drive.google.com/file/d/1vdhrioejsrLEVdfig-h5LbJXDXc6cN67/view?usp=drive_link)**

Files:
- `best_model.pth` — weights with the lowest training loss (use this for generation)
- `last.pth` — final epoch weights + optimizer state (use this to resume training)

After downloading, place them in your Drive folder so the path matches the notebook:
```
/content/drive/MyDrive/asl_diffusion/best_model.pth
```

## Quick Start — Generate Images

Open the notebook in Google Colab (GPU runtime), run the setup cells, then run the
generation cell. To produce all 26 letters:

```python
model = UNet(num_classes).to(device)
model.load_state_dict(torch.load(f"{CKPT_DIR}/best_model.pth", map_location=device))
model.eval()
diffusion = Diffusion(device=device)

letters = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
labels = torch.tensor([dataset.classes.index(L) for L in letters], device=device)
samples = diffusion.sample(model, len(labels), labels)   # -> images in [0, 1]
```

The model is **conditional**, so a label (the letter's class index) is required for
every image.

## Requirements

- Python 3, PyTorch, torchvision
- matplotlib, tqdm, numpy
- kagglehub (for the ASL dataset download)
- A GPU is strongly recommended (training is slow on CPU)

## Notes

- Checkpoints save to Google Drive every epoch, so training survives Colab
  disconnects — re-running the training cell resumes from the last saved epoch.
- At 32×32 the generated signs are recognizable but soft; that's expected for a
  compact diffusion model at this resolution.
