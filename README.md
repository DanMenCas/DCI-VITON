This project implements an end-to-end Virtual Try-On application using Flow-based Warping and Diffusion Models, deployed on Hugging Face Spaces.

**Try the live demo**:
https://huggingface.co/spaces/dmc98/VirtualTryOn_from_scratch

**Download the same used datasets for training**:
https://huggingface.co/dmc98/viton_models

**Preprocessing**

All preprocessing steps are implemented in parsehuman.py.

From each person image, the script extracts:

* DensePose - detectron 2
* Human body segmentation - "yolo12138/segformer-b2-human-parse-24" using transformers module from Hugging Face
* Keypoints with - detectron 2
* Agnostic image (clothing removed)
* Agnostic mask

From the clothing image, the script extracts:

* Cloth mask - rembg

These elements form the conditioning inputs for both the warping and the diffusion stages.

**Flow-Based Warping**

The warping module performs geometric alignment between the clothing item and the target body.

It follows a multi-scale flow prediction architecture with two main inputs:

1. Cloth branch:
  Concatenation of cloth image and cloth mask

2. Body branch:
 Concatenation of DensePose and body segmentation

The architecture is based on:
"High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions"
https://arxiv.org/pdf/2206.14180

The output is a warped cloth aligned to the person’s body shape.

**Diffusion Model**

The images are first encoded through a KL-regularized VAE with latent downsampling factor f = 8, reducing 512×512 images to 64×64 latent tensors.

Then a Noise Scheduler with cosine betas adds noise to:

* The original latent image

* The agnostic latent image with warped cloth

The U-Net receives two separate inputs:

1. Original input concatenates:

    * Noisy original latent
    * Agnostic latent + warped cloth
    * Resized agnostic mask

2. Agnostic input concatenates:

    * Noisy agnostic latent
    * Agnostic latent + warped cloth
    * Resized agnostic mask

The U-Net uses Adaptive Normalization and SE Blocks, and it predicts the added noise.

Training uses two losses:

1. MSE loss:
   
   * Noise prediction from the original input vs. real noise

3. Perceptual loss (VGG19):
   
   * Denoised agnostic noise prediciton vs. the original image

This stage is inspired by:

"Taming the Power of Diffusion Models for High-Quality Virtual Try-On with Appearance Flow"
https://arxiv.org/pdf/2308.06101

After training, a DDIM sampler with 200 steps is used for inference.

**Data**

Flow-based warping trained on 3,000 images
(resized to 256×256, then upsampled to 512×512)

Diffusion model trained on 2,000 warped outputs 
(512x512 and through the latent space 64x64)

The dataset comes from VITON-HD, available here:
https://github.com/shadow2496/VITON-HD

**Opportunities and Future Work**

Due to limited resources, several optimizations were not included:

* No cross-attention mechanisms in the U-Net
* Training performed with batch size = 1
* Dataset limited to 2k samples

Despite this, the model produces promising results and demonstrates that the full pipeline works correctly.

Future improvements include:
* Increasing the dataset to at least 5k–10k samples
* Adding cross-attention to the diffusion network
* Improving the U-Net backbone
* Training with batch size ≥ 10


**Results**

<img width="256" height="512" alt="image" src="https://github.com/user-attachments/assets/0ae36f07-c350-46a4-8fb4-0d124d1f4311" />
<img width="256" height="512" alt="image" src="https://github.com/user-attachments/assets/481bdc94-f756-451a-ab11-95b0c2d6a92c" />
<img width="256" height="512" alt="image" src="https://github.com/user-attachments/assets/a8aa9120-7f72-43e6-aa6a-12adefed5824" />

<img width="256" height="512" alt="image" src="https://github.com/user-attachments/assets/0ae36f07-c350-46a4-8fb4-0d124d1f4311" />
<img width="256" height="512" alt="image" src="https://github.com/user-attachments/assets/ec42dfd8-f095-4e70-a1ff-281c3fb258e0" />
<img width="256" height="512" alt="image" src="https://github.com/user-attachments/assets/490147d0-92a1-4a2e-9d0d-7b578ce4b91e" />

<img width="256" height="512" alt="image" src="https://github.com/user-attachments/assets/852d98fe-fd0d-495e-91ad-1eb3f7a91524" />
<img width="256" height="512" alt="image" src="https://github.com/user-attachments/assets/4464a890-aafc-4765-a480-1e9a5c75ac86" />
<img width="256" height="512" alt="image" src="https://github.com/user-attachments/assets/c61a7bfc-8562-4d24-8fa6-144fff226a16" />




