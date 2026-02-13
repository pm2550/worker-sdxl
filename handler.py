"""
RunPod Serverless handler for Animagine XL 4.0 image generation.

Image delivery: configure BUCKET_ENDPOINT_URL, BUCKET_ACCESS_KEY_ID,
and BUCKET_SECRET_ACCESS_KEY to get image URLs in the response.
Without a bucket, only metadata (seed, dimensions) is returned.
"""

import os
import base64
import io

import torch
from diffusers import StableDiffusionXLPipeline
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


class ModelHandler:
    def __init__(self):
        self.pipe = None
        self.load_models()

    def load_models(self):
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "cagliostrolab/animagine-xl-4.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            add_watermarker=False,
            local_files_only=True,
        ).to("cuda")

        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.enable_model_cpu_offload()


MODELS = ModelHandler()


def _save_and_upload_images(images, job_id):
    """Save images and upload to bucket if configured."""
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    has_bucket = bool(os.environ.get("BUCKET_ENDPOINT_URL"))

    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path)

        if has_bucket:
            image_url = rp_upload.upload_image(job_id, image_path, bucket_name=os.environ.get("BUCKET_NAME", "pm2550"))
            image_urls.append(image_url)

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

    if job_input["seed"] is None:
        job_input["seed"] = int.from_bytes(os.urandom(2), "big")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device).manual_seed(job_input["seed"])

    MODELS.pipe.scheduler = make_scheduler(
        job_input["scheduler"], MODELS.pipe.scheduler.config
    )

    output = MODELS.pipe(
        prompt=job_input["prompt"],
        negative_prompt=job_input["negative_prompt"],
        height=job_input["height"],
        width=job_input["width"],
        num_inference_steps=job_input["num_inference_steps"],
        guidance_scale=job_input["guidance_scale"],
        num_images_per_prompt=job_input["num_images"],
        generator=generator,
    ).images

    image_urls = _save_and_upload_images(output, job["id"])

    results = {"seed": job_input["seed"]}

    if image_urls:
        results["images"] = image_urls
        results["image_url"] = image_urls[0]
    else:
        results["image_count"] = len(output)

    return results


runpod.serverless.start({"handler": generate_image})
