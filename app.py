# app.py

print("app.py is already running!")
import os
print("os is already imported!")
import torch
print("torch is already imported!")
import gradio as gr
print("gradio is already imported!")
from PIL import Image
print("PIL is already imported!")

os.environ["HF_HOME"] = "/tmp"

print("Starting up... CUDA available:", torch.cuda.is_available())

# ←←←←←←←←←←←←←←←←←←←←←← GLOBAL VARIABLES = NONE ←←←←←←←←←←←←←←←←←←←←←←
viton_pipeline = None
vae = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def lazy_init():
    global viton_pipeline, vae
    if viton_pipeline is not None:
        return  # already initialized

    print("First request → loading models (this takes 1–3 minutes)...")
    from huggingface_hub import hf_hub_download
    from diffusers import AutoencoderKL
    from viton_pipeline import VitonPipeline

    checkpoint_ddpm_path = hf_hub_download(
        repo_id="dmc98/viton_models",
        filename="latent512_1500img_resnet_viton_adamW_schedulelr1e-05_vgg0.001_epoch_300.pth"
    )
    checkpoint_warping_path = hf_hub_download(
        repo_id="dmc98/viton_models",
        filename="3000imgs_warping_lr5e-05_vgg0.1_tvlmabda_1epoch_30.pth"
    )

    checkpoint_ddpm = torch.load(checkpoint_ddpm_path, map_location=device, weights_only=True)
    checkpoint_warping = torch.load(checkpoint_warping_path, map_location=device, weights_only=True)

    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
    vae = vae.to(device)

    viton_pipeline = VitonPipeline((512, 512), checkpoint_ddpm, checkpoint_warping, vae)
    print("Models loaded successfully!")

def tensor_as_image(tensor):
    if tensor.shape != (1, 3, 512, 512):
        raise ValueError(f"Bad shape {tensor.shape}")
    tensor = tensor.squeeze(0)
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    tensor = (tensor * 255).astype("uint8")
    return Image.fromarray(tensor)

def process_images(person_img: Image.Image, cloth_img: Image.Image):
    lazy_init()                               # ← this is the only important line
    output_tensor = viton_pipeline(person_img, cloth_img)
    return tensor_as_image(output_tensor)

# Gradio interface
iface = gr.Interface(
    fn=process_images,
    inputs=[
        gr.Image(type="pil", label="Person (full body, straight pose)"),
        gr.Image(type="pil", label="Cloth (front-facing)"),
    ],
    outputs=gr.Image(type="pil", label="Virtual Try-On Result"),
    title="VITON Virtual Try-On",
    description="First request will take 1–4 minutes to load the models. Subsequent requests are fast.",
    allow_flagging="never",
)

if __name__ == "__main__":
    iface.launch()