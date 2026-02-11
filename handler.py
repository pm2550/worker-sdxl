import os
import atexit
import base64
import time

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


def _log(msg):
    print(f"[handler] {msg}", flush=True)


atexit.register(lambda: _log("Python process exiting (atexit)"))

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
diffusers_logging.disable_progress_bar()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

LOCAL_FILES_ONLY_ENV = "LOCAL_FILES_ONLY"

_log(f"Module loaded. CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    _log(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def _local_files_only():
    return os.environ.get(LOCAL_FILES_ONLY_ENV, "1").strip() == "1"


# --- Scheduler helpers (only instantiate the one we need) ---

SCHEDULER_MAP = {
    "PNDM": PNDMScheduler,
    "KLMS": LMSDiscreteScheduler,
    "DDIM": DDIMScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "DPMSolverSinglestep": DPMSolverSinglestepScheduler,
}


def make_scheduler(name, config):
    scheduler_cls = SCHEDULER_MAP.get(name, DDIMScheduler)
    return scheduler_cls.from_config(config)


# --- Model loading (eager, at module level, matching official RunPod worker) ---

class ModelHandler:
    def __init__(self):
        self.base = None
        self.refiner = None
        self.load_base_model()

    def load_base(self):
        local_files_only = _local_files_only()
        _log(f"Loading VAE (local_files_only={local_files_only})...")
        t0 = time.time()
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
            local_files_only=local_files_only,
        )
        _log(f"VAE loaded in {time.time() - t0:.1f}s")

        _log("Loading SDXL base pipeline...")
        t0 = time.time()
        base_pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            add_watermarker=False,
            local_files_only=local_files_only,
        ).to("cuda")
        _log(f"SDXL base loaded to GPU in {time.time() - t0:.1f}s")

        base_pipe.enable_vae_slicing()
        base_pipe.set_progress_bar_config(disable=True)

        return base_pipe

    def load_refiner(self):
        local_files_only = _local_files_only()
        _log("Loading refiner pipeline...")
        t0 = time.time()
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
        _log(f"Refiner loaded to GPU in {time.time() - t0:.1f}s")

        refiner_pipe.enable_vae_slicing()
        refiner_pipe.set_progress_bar_config(disable=True)

        return refiner_pipe

    def load_base_model(self):
        _log("load_base_model() called")
        self.base = self.load_base()
        _log("load_base_model() done")

    def get_refiner(self):
        if self.refiner is None:
            self.refiner = self.load_refiner()
        return self.refiner


_log("Eagerly loading models at startup...")
t_model_start = time.time()
MODELS = ModelHandler()
_log(f"Models loaded in {time.time() - t_model_start:.1f}s")


# --- Helpers ---

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


# --- Handler ---

@torch.inference_mode()
def generate_image(job):
    job_id = job.get("id", "unknown")
    _log(f"generate_image called, job_id={job_id}")
    t_start = time.time()

    try:
        job_input = job.get("input", {})
        validated_input = validate(job_input, INPUT_SCHEMA)

        if "errors" in validated_input:
            _log(f"Validation errors: {validated_input['errors']}")
            return {"error": validated_input["errors"]}

        job_input = validated_input["validated_input"]

        seed = job_input["seed"]
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
            job_input["seed"] = seed

        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device).manual_seed(seed)

        MODELS.base.scheduler = make_scheduler(
            job_input["scheduler"], MODELS.base.scheduler.config
        )

        starting_image = job_input.get("image_url")
        use_starting_image = bool(starting_image)

        if use_starting_image:
            _log("Running img2img with refiner...")
            refiner = MODELS.get_refiner()
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
            _log(f"Running txt2img: {job_input['width']}x{job_input['height']}, "
                 f"steps={job_input['num_inference_steps']}, seed={seed}")
            base_result = MODELS.base(
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

        _log(f"Inference done, {len(output)} image(s) generated")

        image_urls = _save_and_upload_images(output, job_id)
        _log(f"Images saved, {len(image_urls)} url(s), "
             f"first url type: {'base64' if image_urls[0].startswith('data:') else 'url'}")

        results = {
            "images": image_urls,
            "image_url": image_urls[0],
            "seed": job_input["seed"],
        }

        if use_starting_image:
            results["refresh_worker"] = True

        elapsed = time.time() - t_start
        _log(f"SUCCESS in {elapsed:.1f}s, returning keys={list(results.keys())}")
        return results

    except Exception as err:
        elapsed = time.time() - t_start
        _log(f"ERROR in {elapsed:.1f}s: {type(err).__name__}: {err}")
        return {
            "error": f"{type(err).__name__}: {err}",
            "refresh_worker": True,
        }


_log("Starting RunPod serverless worker...")
runpod.serverless.start({"handler": generate_image})
