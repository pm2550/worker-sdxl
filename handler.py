"""
RunPod Serverless handler for SDXL image generation.

Image delivery: configure BUCKET_ENDPOINT_URL, BUCKET_ACCESS_KEY_ID,
and BUCKET_SECRET_ACCESS_KEY to get image URLs in the response.
Without a bucket, only metadata (seed, dimensions) is returned.
"""

import os
import io
import uuid
import logging

import boto3
from botocore.config import Config as BotoConfig

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
from runpod.serverless.utils.rp_validator import validate

logger = logging.getLogger("handler")

from schemas import INPUT_SCHEMA

torch.cuda.empty_cache()


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


def _get_s3_client():
    """Create boto3 S3 client for R2/S3-compatible storage."""
    endpoint_url = os.environ.get("BUCKET_ENDPOINT_URL")
    access_key = os.environ.get("BUCKET_ACCESS_KEY_ID")
    secret_key = os.environ.get("BUCKET_SECRET_ACCESS_KEY")
    if not all([endpoint_url, access_key, secret_key]):
        return None
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=BotoConfig(s3={"addressing_style": "path"}, signature_version="s3v4"),
        region_name="auto",
    )


def _upload_to_bucket(image, job_id, index):
    """Upload a PIL image to S3/R2 bucket, return public URL."""
    s3 = _get_s3_client()
    if not s3:
        return None
    bucket_name = os.environ.get("BUCKET_NAME", "pm2550")
    key = f"{job_id}/{index}.png"
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    s3.put_object(Bucket=bucket_name, Key=key, Body=buf.getvalue(), ContentType="image/png")
    endpoint_url = os.environ.get("BUCKET_ENDPOINT_URL").rstrip("/")
    return f"{endpoint_url}/{bucket_name}/{key}"


def _save_and_upload_images(images, job_id):
    """Upload images to bucket if configured, return URLs."""
    image_urls = []
    for index, image in enumerate(images):
        try:
            url = _upload_to_bucket(image, job_id, index)
            if url:
                image_urls.append(url)
        except Exception as e:
            logger.error(f"Upload failed for image {index}: {e}")
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

    results = {"seed": job_input["seed"]}

    if image_urls:
        results["images"] = image_urls
        results["image_url"] = image_urls[0]
    else:
        results["image_count"] = len(output)

    if starting_image:
        results["refresh_worker"] = True

    return results


runpod.serverless.start({"handler": generate_image})
