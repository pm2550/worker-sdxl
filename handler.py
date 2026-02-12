"""
RunPod Serverless handler for SDXL image generation.
"""

import os
import sys
import io
import base64
import json
import time

import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    AutoencoderKL,
)
from diffusers.utils import load_image

from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverSinglestepScheduler,
)

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from schemas import INPUT_SCHEMA

torch.cuda.empty_cache()

# Maximum base64 response size (bytes) for inline delivery.
# RunPod's result delivery pipeline has undocumented size limits;
# responses over ~100KB fail silently. Use bucket upload for large images.
MAX_INLINE_B64_SIZE = 75_000


class ModelHandler:
    def __init__(self):
        self.base = None
        self.refiner = None
        self.load_models()

    def load_base(self):
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        base_pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            add_watermarker=False,
            local_files_only=True,
        ).to("cuda")

        base_pipe.enable_xformers_memory_efficient_attention()
        base_pipe.enable_model_cpu_offload()

        return base_pipe

    def load_refiner(self):
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            add_watermarker=False,
            local_files_only=True,
        ).to("cuda")

        refiner_pipe.enable_xformers_memory_efficient_attention()
        refiner_pipe.enable_model_cpu_offload()

        return refiner_pipe

    def load_models(self):
        self.base = self.load_base()
        self.refiner = self.load_refiner()


MODELS = ModelHandler()


def _image_to_b64(image, quality=80):
    """Convert PIL image to base64 JPEG string."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}", len(b64)


def _save_and_upload_images(images, job_id):
    """
    Save images and return URLs.

    With BUCKET_ENDPOINT_URL configured: uploads to S3-compatible bucket, returns URLs.
    Without bucket: returns base64 JPEG inline (auto-compressed to fit delivery limits).
    """
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    has_bucket = bool(os.environ.get("BUCKET_ENDPOINT_URL"))

    for index, image in enumerate(images):
        if has_bucket:
            image_path = os.path.join(f"/{job_id}", f"{index}.png")
            image.save(image_path)
            image_url = rp_upload.upload_image(job_id, image_path)
            image_urls.append(image_url)
        else:
            # No bucket: try inline base64 with progressive compression
            for q in (80, 60, 40):
                b64_url, b64_len = _image_to_b64(image, quality=q)
                if b64_len <= MAX_INLINE_B64_SIZE:
                    image_urls.append(b64_url)
                    break
            else:
                # Still too large - resize and compress aggressively
                thumb = image.copy()
                thumb.thumbnail((256, 256))
                b64_url, _ = _image_to_b64(thumb, quality=60)
                image_urls.append(b64_url)

    rp_cleanup.clean([f"/{job_id}"])
    return image_urls


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
        "DPMSolverSinglestep": DPMSolverSinglestepScheduler.from_config(config),
    }[name]


@torch.inference_mode()
def generate_image(job):
    job_input = job["input"]

    validated_input = validate(job_input, INPUT_SCHEMA)

    if "errors" in validated_input:
        return {"error": validated_input["errors"]}
    job_input = validated_input["validated_input"]

    starting_image = job_input["image_url"]

    if job_input["seed"] is None:
        job_input["seed"] = int.from_bytes(os.urandom(2), "big")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device).manual_seed(job_input["seed"])

    MODELS.base.scheduler = make_scheduler(
        job_input["scheduler"], MODELS.base.scheduler.config
    )

    if starting_image:
        init_image = load_image(starting_image).convert("RGB")
        output = MODELS.refiner(
            prompt=job_input["prompt"],
            num_inference_steps=job_input["refiner_inference_steps"],
            strength=job_input["strength"],
            image=init_image,
            generator=generator,
        ).images
    else:
        image = MODELS.base(
            prompt=job_input["prompt"],
            negative_prompt=job_input["negative_prompt"],
            height=job_input["height"],
            width=job_input["width"],
            num_inference_steps=job_input["num_inference_steps"],
            guidance_scale=job_input["guidance_scale"],
            denoising_end=job_input["high_noise_frac"],
            output_type="latent",
            num_images_per_prompt=job_input["num_images"],
            generator=generator,
        ).images

        output = MODELS.refiner(
            prompt=job_input["prompt"],
            num_inference_steps=job_input["refiner_inference_steps"],
            strength=job_input["strength"],
            image=image,
            num_images_per_prompt=job_input["num_images"],
            generator=generator,
        ).images

    image_urls = _save_and_upload_images(output, job["id"])

    results = {
        "images": image_urls,
        "image_url": image_urls[0],
        "seed": job_input["seed"],
    }

    if starting_image:
        results["refresh_worker"] = True

    return results


runpod.serverless.start({"handler": generate_image})
