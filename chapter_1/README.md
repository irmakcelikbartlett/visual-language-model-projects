# Chapter 1 labs: Introduction to Vision and Language

These notebooks rebuild Chapter 1 of *Vision Language Models* as runnable labs. They use original explanations, not copied book text. Run them in order.

| Order | Notebook | What you learn | GPU? | HF token? |
| --- | --- | --- | --- | --- |
| 1 | `01_pixels_filters_cnns.ipynb` | Pixels, DCT, convolution, Sobel/Prewitt, keypoints, CNN / ResNet skip / U-Net | No | No |
| 2 | `02_from_rnn_to_vit_clip.ipynb` | Tokens, RNN vs transformer, ViT, CLIP zero-shot, Hugging Face Hub / datasets | Yes, helpful | No |
| 3 | `03_text_to_image_search.ipynb` | SigLIP embeddings + FAISS text-to-image search on WikiArt | Yes | Yes |

## Open in Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com).
2. **File → Upload notebook** and choose one of the `.ipynb` files (or upload the whole `chapter_1` folder to Drive and open from there).
3. For notebooks 02 and 03: **Runtime → Change runtime type → T4 GPU** (or any GPU).
4. Run cells from top to bottom. The first code cell installs packages.

Colab does not keep a GPU unless you pick one. If you forget, the notebooks fall back to CPU; CLIP/SigLIP will just be slower.

## Hugging Face token (notebook 03 only)

The WikiArt rows endpoint used in the search app may require a logged-in Hugging Face account.

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Create a read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. In Colab: click the key icon (Secrets) → add `HF_TOKEN` → enable notebook access.
4. The notebook reads `userdata.get("HF_TOKEN")` and calls `huggingface_hub.login`.

If Secrets are unavailable, the notebook also accepts `os.environ["HF_TOKEN"]`.

## Colab checklist after you run everything

Use this as a “did I actually learn it?” pass, not as a grade.

**Notebook 01**

- [ ] DCT: keeping more coefficients makes the reconstruction sharper; dropping high frequencies blurs edges and texture.
- [ ] 1D kernel `[-1, 1]` lights up where the signal jumps.
- [ ] Sobel/Prewitt maps highlight object boundaries on the sample image.
- [ ] SIFT matches corresponding points between a crop and the full image (some mismatches are normal).
- [ ] The tiny CNN’s first-layer filters stop looking random after a few training steps.
- [ ] Residual add: `out = x + F(x)` keeps a copy of the input; U-Net concat brings encoder detail back to the decoder.

**Notebook 02**

- [ ] ViT `pipeline("image-classification")` on the bee image returns insect/flower-like ImageNet labels.
- [ ] CLIP scores the matching text label higher than distractors for the same image.
- [ ] `load_dataset("merve/vqav2-small")` prints splits and a sample question/image.

**Notebook 03**

- [ ] FAISS index size equals the embedding dimension printed from the SigLIP model (do not hard-code 1152).
- [ ] A query like `a portrait` or `a landscape` returns paintings that roughly match the text.
- [ ] Changing the query changes which images come back.

## Local Mac note

Notebook 01 can run locally with numpy, matplotlib, OpenCV, and PyTorch. Notebooks 02–03 are sized for Colab. If you run them on a Mac, use CPU/`faiss-cpu` and expect slower downloads.
