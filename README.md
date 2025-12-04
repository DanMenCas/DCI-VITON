# Virtual Try On from scratch



Designed, implemented, and deployed an end-to-end Virtual Try-On application on Hugging Face Spaces using **Flow based Warping** and **Difussion Models.** You can use the app here:

https://huggingface.co/spaces/dmc98/VirtualTryOn\_from\_scratch

for training purposes there are two jupyter notebooks, the data can be downloaded from there:

https://huggingface.co/dmc98/viton_models

**Preprocessing**


In the parsehuman.py file you can see all the preprocessing part of the model, it extracts from an person image, densepose, body segmentation, key points to create the agnostic image and agnostic mask and from the cloth image just the cloth mask.



**Flow based Warping**



The warping process is a geometric alignment that use a multi-scale flow prediction, there are two inputs mainly is one for the cloth and the other for the body, the first is a concatenation of cloth and cloth mask, and the second one is the densepose and the body segmentation, the structure is main based in the paper "High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions" **https://arxiv.org/pdf/2206.14180.**



**Difussion Model**



Tehe first step is to pass the images through a KL-regularized autoencoder with latent-space with downsampling factor 𝑓 = 8, so from a size 512x512 to 64x64, after that a Noise Scheduler adds noise to the output of the warping module, then, it pass through a Unet network, that has two inputs, the first one is the concatenation of the latent noised original image, the latent agnostic image with the warped cloth and the resized agnostic mask, the second one, is the same but instead the latent noised original image is latent noised agnostic image with warped cloth. The Unet itself uses Adaptative normalization and SE Blocks and it predicts the noise added to the image, for this noise prediction, MSE loss is used with the output with first input, and for perceptual loss, VGG19 layers are used to compare the output of the second input with the original image. Part of the process is based in the "Taming the Power of Diffusion Models for High-Quality Virtual Try-On with Appearance Flow" **https://arxiv.org/pdf/2308.06101**



When the diffusion model is trained, then a Diffusion denoising implicit model (DDIM) is applied with 200 steps.



**Data**



The warping process was trained with 3000 images resized to 256x256 and the output is resized again to 512x512, then the Diffusion model was trained only with 2k images of the warping outputs. The training dataset is the VITON-HD and you can find it through its GitHub, **https://github.com/shadow2496/VITON-HD.**



**Opportunities**



Mainly due to the lack of resources cross attention mechanism was not added to the diffusion network and the process only was trained with batch = 1, however the results are not bad, and it means that the whole model works as expected, with more resources teh goal is to increase the data from 2k to at least 5k images, improve Unet network and add cross attention mechanism and increases the batch size to at least 10.

**Results**

<img width="256" height="512" alt="image" src="https://github.com/user-attachments/assets/0ae36f07-c350-46a4-8fb4-0d124d1f4311" /><img width="256" height="512" alt="image" src="https://github.com/user-attachments/assets/481bdc94-f756-451a-ab11-95b0c2d6a92c" /><img width="256" height="512" alt="image" src="https://github.com/user-attachments/assets/fed6625d-d7ef-49bf-99c4-c583e3f54982" />






