import os
import sys
import base64
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


def _log(msg):
    print(f"[handler] {msg}", flush=True)


_log(f"Python {sys.version}")
_log(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    _log(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()


class ModelHandler:
    def __init__(self):
        self.base = None
        self.refiner = None
        self.load_models()

    def load_base(self):
        _log("Loading VAE for base...")
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        _log("Loading base pipeline...")
        t0 = time.time()
        base_pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            add_watermarker=False,
            local_files_only=True,
        ).to("cuda")
        _log(f"Base pipeline loaded in {time.time() - t0:.1f}s")

        _log("Enabling xformers for base...")
        base_pipe.enable_xformers_memory_efficient_attention()
        _log("Enabling CPU offload for base...")
        base_pipe.enable_model_cpu_offload()
        _log("Base model ready")
        return base_pipe

    def load_refiner(self):
        _log("Loading VAE for refiner...")
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        _log("Loading refiner pipeline...")
        t0 = time.time()
        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            add_watermarker=False,
            local_files_only=True,
        ).to("cuda")
        _log(f"Refiner pipeline loaded in {time.time() - t0:.1f}s")

        _log("Enabling xformers for refiner...")
        refiner_pipe.enable_xformers_memory_efficient_attention()
        _log("Enabling CPU offload for refiner...")
        refiner_pipe.enable_model_cpu_offload()
        _log("Refiner model ready")
        return refiner_pipe

    def load_models(self):
        self.base = self.load_base()
        self.refiner = self.load_refiner()


_log("Loading models...")
t_start = time.time()
MODELS = ModelHandler()
_log(f"All models loaded in {time.time() - t_start:.1f}s")


def _save_and_upload_images(images, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path)

        if os.environ.get("BUCKET_ENDPOINT_URL", False):
            image_url = rp_upload.upload_image(job_id, image_path)
            image_urls.append(image_url)
        else:
            jpg_path = image_path.replace(".png", ".jpg")
            image.save(jpg_path, format="JPEG", quality=85)
            with open(jpg_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                image_urls.append(f"data:image/jpeg;base64,{image_data}")

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
    job_id = job.get("id", "unknown")
    _log(f"generate_image called, job_id={job_id}")
    t_start = time.time()

    job_input = job["input"]

    validated_input = validate(job_input, INPUT_SCHEMA)

    if "errors" in validated_input:
        _log(f"Validation errors: {validated_input['errors']}")
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
        _log("Running img2img with refiner...")
        init_image = load_image(starting_image).convert("RGB")
        output = MODELS.refiner(
            prompt=job_input["prompt"],
            num_inference_steps=job_input["refiner_inference_steps"],
            strength=job_input["strength"],
            image=init_image,
            generator=generator,
        ).images
    else:
        _log(f"Running txt2img: {job_input['width']}x{job_input['height']}, "
             f"steps={job_input['num_inference_steps']}, seed={job_input['seed']}")

        _log("Running base pipeline (latent output)...")
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
        _log("Base pipeline done, running refiner...")

        output = MODELS.refiner(
            prompt=job_input["prompt"],
            num_inference_steps=job_input["refiner_inference_steps"],
            strength=job_input["strength"],
            image=image,
            num_images_per_prompt=job_input["num_images"],
            generator=generator,
        ).images
        _log(f"Refiner done, {len(output)} image(s)")

    image_urls = _save_and_upload_images(output, job["id"])

    import json
    results = {
        "images": image_urls,
        "image_url": image_urls[0],
        "seed": job_input["seed"],
    }
    result_size = len(json.dumps(results))
    _log(f"Response: {result_size} bytes ({result_size/1024:.1f} KB)")

    if starting_image:
        results["refresh_worker"] = True

    elapsed = time.time() - t_start
    _log(f"SUCCESS in {elapsed:.1f}s")
    return results


_log("Starting RunPod serverless worker...")
runpod.serverless.start({"handler": generate_image})
