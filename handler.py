import os
import base64

import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    AutoencoderKL,
)
from diffusers.utils import load_image
from diffusers.utils import logging as diffusers_logging

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

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
diffusers_logging.disable_progress_bar()

torch.cuda.empty_cache()

LOCAL_FILES_ONLY_ENV = "LOCAL_FILES_ONLY"


class ModelHandler:
    def __init__(self):
        self.base = None
        self.refiner = None
        self.load_base_model()

    def load_base(self):
        local_files_only = _local_files_only()
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
            local_files_only=local_files_only,
        )
        base_pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            add_watermarker=False,
            local_files_only=local_files_only,
        ).to("cuda")

        base_pipe.enable_xformers_memory_efficient_attention()
        base_pipe.enable_vae_slicing()
        base_pipe.set_progress_bar_config(disable=True)

        return base_pipe

    def load_refiner(self):
        local_files_only = _local_files_only()
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
            local_files_only=local_files_only,
        )
        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            add_watermarker=False,
            local_files_only=local_files_only,
        ).to("cuda")

        refiner_pipe.enable_xformers_memory_efficient_attention()
        refiner_pipe.enable_vae_slicing()
        refiner_pipe.set_progress_bar_config(disable=True)

        return refiner_pipe

    def load_base_model(self):
        self.base = self.load_base()

    def get_refiner(self):
        if self.refiner is None:
            self.refiner = self.load_refiner()
        return self.refiner


MODELS = None


def _get_models():
    global MODELS
    if MODELS is None:
        MODELS = ModelHandler()
    return MODELS


def _job_tmp_dir(job_id):
    return os.path.join("/tmp", str(job_id))


def _save_and_upload_images(images, job_id):
    output_dir = _job_tmp_dir(job_id)
    os.makedirs(output_dir, exist_ok=True)

    image_urls = []
    for index, image in enumerate(images):
        image_path = os.path.join(output_dir, f"{index}.png")
        image.save(image_path)

        if os.environ.get("BUCKET_ENDPOINT_URL", False):
            image_url = rp_upload.upload_image(job_id, image_path)
            image_urls.append(image_url)
        else:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                image_urls.append(f"data:image/png;base64,{image_data}")

    rp_cleanup.clean([output_dir])
    return image_urls


def make_scheduler(name, config):
    scheduler_map = {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
        "DPMSolverSinglestep": DPMSolverSinglestepScheduler.from_config(config),
    }
    return scheduler_map.get(name, DDIMScheduler.from_config(config))


def _local_files_only():
    return os.environ.get(LOCAL_FILES_ONLY_ENV, "1").strip() == "1"


@torch.inference_mode()
def generate_image(job):
    job_input = job.get("input", {})
    validated_input = validate(job_input, INPUT_SCHEMA)

    if "errors" in validated_input:
        return {"error": validated_input["errors"]}

    job_input = validated_input["validated_input"]

    seed = job_input["seed"]
    if seed is None:
        seed = int.from_bytes(os.urandom(2), "big")
        job_input["seed"] = seed

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device).manual_seed(seed)

    models = _get_models()
    models.base.scheduler = make_scheduler(
        job_input["scheduler"], models.base.scheduler.config
    )

    starting_image = job_input.get("image_url")
    use_starting_image = bool(starting_image)

    try:
        if use_starting_image:
            refiner = models.get_refiner()
            init_image = load_image(starting_image).convert("RGB")
            refiner_result = refiner(
                prompt=job_input["prompt"],
                num_inference_steps=job_input["refiner_inference_steps"],
                strength=job_input["strength"],
                image=init_image,
                generator=generator,
            )
            output = refiner_result.images
        else:
            # Fast path for normal txt2img:
            # use SDXL base directly and skip refiner.
            base_result = models.base(
                prompt=job_input["prompt"],
                negative_prompt=job_input["negative_prompt"],
                height=job_input["height"],
                width=job_input["width"],
                num_inference_steps=job_input["num_inference_steps"],
                guidance_scale=job_input["guidance_scale"],
                output_type="pil",
                num_images_per_prompt=job_input["num_images"],
                generator=generator,
            )
            output = base_result.images

    except RuntimeError as err:
        return {
            "error": f"RuntimeError: {err}",
            "refresh_worker": True,
        }
    except Exception as err:
        return {
            "error": f"Unexpected error: {err}",
            "refresh_worker": True,
        }

    image_urls = _save_and_upload_images(output, job["id"])

    results = {
        "images": image_urls,
        "image_url": image_urls[0],
        "seed": job_input["seed"],
    }

    if use_starting_image:
        results["refresh_worker"] = True

    return results


runpod.serverless.start({"handler": generate_image})
