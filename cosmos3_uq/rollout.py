"""Cosmos3 forward-dynamics rollouts with the velocity closure captured at the query noise level.

Backend: the diffusers Cosmos3OmniPipeline. Sampling settings: UniPC, 30 steps, flow shift 10, guidance 1.0 (the
closure is then the pure conditional field), chunk of 20 actions = 21 frames, 9D AV actions. Cosmos3-Nano runs at the
480 resolution tier; Cosmos3-Edge also runs at the 256 tier, which is small enough for exact diagonals (exact.py).

The pipeline packs [text | vision | action] into one joint sequence and calls the transformer once per step with
keyword arguments only. `Capture` wraps that call: at the sampler step closest to the requested sigma it stores every
argument except the noisy vision tokens, and a step-end callback stores the float32 latents entering that step and the
final latents. The closure v(x) re-runs the transformer with x in place of the vision tokens and zeroes the
conditioning frame, exactly as the pipeline does before its scheduler step.
"""
import copy
import json
import time

import numpy as np
import torch
from PIL import Image

MODELS = {"nano": "nvidia/Cosmos3-Nano", "edge": "nvidia/Cosmos3-Edge"}
AV_PROMPT = "You are an autonomous vehicle planning system."
SAMPLING = {"num_steps": 30, "guidance": 1.0, "shift": 10.0}


def resolve_model(model):
    """Accept a short alias ("nano", "edge") or a Hugging Face model id."""
    return MODELS.get(model, model)


def load_pipe(model="nano", device="cuda"):
    """Load the pipeline in bf16 without the guardrail and with the shift-10 UniPC schedule.

    The checkpoint config sets shift 1 with Karras sigmas, which has to be overridden to match the reference sampling.
    """
    from diffusers import Cosmos3OmniPipeline, UniPCMultistepScheduler

    pipe = Cosmos3OmniPipeline.from_pretrained(
        resolve_model(model), torch_dtype=torch.bfloat16, safety_checker=None, enable_safety_checker=False
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=SAMPLING["shift"], use_karras_sigmas=False
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def sigma_grid(pipe, num_steps):
    """The sampler's sigma grid (length num_steps + 1, last entry 0) and its timesteps.

    Mirrors the pipeline: models whose config sets use_native_flow_schedule (Cosmos3-Edge) pass an explicit linear
    sigma list to the scheduler, which moves the last step from sigma 0.262 to 0.256.
    """
    sched = copy.deepcopy(pipe.scheduler)
    if pipe.config.get("use_native_flow_schedule", False):
        sigmas = np.linspace(1.0 - 1.0 / sched.config.num_train_timesteps, 0.0, num_steps + 1)[:-1]
        sched.set_timesteps(num_steps, sigmas=sigmas)
    else:
        sched.set_timesteps(num_steps)
    return sched.sigmas.cpu().numpy().astype(np.float64), sched.timesteps.cpu().numpy().astype(np.float64)


def query_step(sigmas, sigma_q):
    """Index of the sampler step whose input sigma is closest to the requested one.

    The grid, not sigma_q, sets the capture point, so the captured sigma must be recorded with the results.
    """
    return int(np.argmin(np.abs(sigmas[:-1] - sigma_q)))


def load_actions(path, chunk_size):
    """Raw domain actions [T, D] from a JSON list or a .npy, truncated to the chunk.

    The pipeline repeats the last action when the sequence is shorter than the chunk.
    """
    if str(path).endswith(".npy"):
        a = np.load(path)
    else:
        a = np.asarray(json.load(open(path)), dtype=np.float32)
    return torch.as_tensor(a[:chunk_size], dtype=torch.float32)


class Capture:
    """Records the conditional transformer call at one sampler step, the float32 latents entering it and the final
    latents of the rollout."""

    def __init__(self, pipe, step_index, timestep, num_steps):
        self.pipe = pipe
        self.step_index = step_index
        self.timestep = float(timestep)
        self.num_steps = num_steps
        self.kwargs = None
        self.x = None
        self.x_bf16 = None
        self.final = None

    def install(self):
        self._orig = self.pipe.transformer.forward
        self.pipe.transformer.forward = self._forward

    def remove(self):
        # Deleting the instance attribute restores the class method.
        del self.pipe.transformer.forward

    def _forward(self, *args, **kw):
        ts = kw.get("vision_timesteps")
        if self.kwargs is None and ts is not None and abs(float(ts.reshape(-1)[0]) - self.timestep) < 0.5:
            self.kwargs = {k: v for k, v in kw.items() if k not in ("vision_tokens", "return_dict")}
            self.x_bf16 = kw["vision_tokens"][0].detach().clone()
        return self._orig(*args, **kw)

    def callback(self, pipe, i, t, cb):
        """Step-end callback: the latents after step i are the input of step i + 1; the last ones are the sample."""
        if i == self.step_index - 1:
            self.x = cb["latents"].detach().clone().float()
        if i == self.num_steps - 1:
            self.final = cb["latents"].detach().clone().float()
        return {}


def make_closure(pipe, cap):
    """v(x) with every other input frozen; the conditioning frame's velocity is zeroed like in the pipeline.

    Returns (v_fn, mask over latent frames, sorted indices of the noisy latent frames).
    """
    transformer = pipe.transformer
    dtype = transformer.dtype
    x_shape = cap.x.shape
    noisy = cap.kwargs["vision_noisy_frame_indexes"][0].tolist()
    mask = torch.zeros(x_shape[2], device=cap.x.device, dtype=torch.float32)
    mask[noisy] = 1.0
    mask = mask.view(1, 1, -1, 1, 1)

    def v_fn(x):
        preds, _, _ = transformer(vision_tokens=[x.to(dtype)], return_dict=False, **cap.kwargs)
        return preds[0].float().reshape(x_shape) * mask

    return v_fn, mask, sorted(noisy)


def decode_latents(pipe, latents):
    """Normalized vision latents [1, C, T, H, W] -> list of PIL frames, through the pipeline's own decode path."""
    dtype = pipe.vae.dtype
    mean = pipe._vae_latents_mean.to(device=latents.device, dtype=dtype)
    inv_std = pipe._vae_latents_inv_std.to(device=latents.device, dtype=dtype)
    z = latents.to(dtype) / inv_std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
    with torch.no_grad():
        video = pipe.vae.decode(z).sample.float()
    return pipe.video_processor.postprocess_video(video, output_type="pil")[0]


@torch.no_grad()
def encode_video_latents(pipe, frames, chunk_size, resolution_tier, device="cuda"):
    """Real frames uint8 [T >= chunk_size + 1, H, W, 3] -> normalized latents [1, C, T_l, h, w].

    Uses the pipeline's own action-video preprocessing (resolution bin, padding) and VAE path, so the result lines up
    cell by cell with the rollout's final latents. Frame 0 must be the conditioning frame.
    """
    if len(frames) < chunk_size + 1:
        raise ValueError(f"need at least {chunk_size + 1} frames, got {len(frames)}")
    pil = [Image.fromarray(np.asarray(f)) for f in frames[: chunk_size + 1]]
    vt, image_size, _, _ = pipe._prepare_action_video_conditioning(
        pil, resolution_tier, chunk_size + 1, device=device, dtype=torch.float32
    )
    z = pipe._encode_video(vt).float()
    return pipe._remove_action_video_padding_from_latent(z, image_size).contiguous()


def action_condition(record):
    """The pipeline's forward-dynamics condition for a record. Returns (condition, conditioning image, chunk size).

    Record keys: image, actions, and optionally chunk_size (20), domain ("av"), resolution_tier (480),
    view_point ("ego_view").
    """
    from diffusers import CosmosActionCondition

    chunk = int(record.get("chunk_size", 20))
    image = Image.open(record["image"]).convert("RGB")
    action = CosmosActionCondition(
        mode="forward_dynamics",
        chunk_size=chunk,
        domain_name=record.get("domain", "av"),
        resolution_tier=int(record.get("resolution_tier", 480)),
        raw_actions=load_actions(record["actions"], chunk),
        image=image,
        view_point=record.get("view_point", "ego_view"),
    )
    return action, image, chunk


def rollout(pipe, record, sigma_q=0.3, num_steps=SAMPLING["num_steps"], guidance=SAMPLING["guidance"], seed=0):
    """Run one forward-dynamics record and capture the closure inputs.

    Record keys: those of `action_condition`, plus optionally fps (10) and prompt.
    """
    device = pipe._get_execution_device()
    action, image, chunk = action_condition(record)
    sigmas, timesteps = sigma_grid(pipe, num_steps)
    i_q = query_step(sigmas, sigma_q)
    cap = Capture(pipe, i_q, timesteps[i_q], num_steps)
    cap.install()
    generator = torch.Generator(device=device).manual_seed(seed)
    t0 = time.time()
    try:
        result = pipe(
            prompt=record.get("prompt", AV_PROMPT),
            action=action,
            fps=float(record.get("fps", 10)),
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            use_system_prompt=False,
            generator=generator,
            callback_on_step_end=cap.callback,
            callback_on_step_end_tensor_inputs=["latents"],
            enable_safety_check=False,
        )
    finally:
        cap.remove()
    seconds = time.time() - t0
    if cap.kwargs is None:
        raise RuntimeError("the transformer call at the query step was not captured")
    if cap.x is None:
        # Step 0: the input is the initial noise, and the callback never fires before step 0.
        cap.x = cap.x_bf16.float()
    return {
        "frames": result.video,
        "cap": cap,
        "sigma_requested": float(sigma_q),
        "sigma": float(sigmas[i_q]),
        "step_index": i_q,
        "num_steps": num_steps,
        "timestep": float(timesteps[i_q]),
        "guidance": float(guidance),
        "shift": SAMPLING["shift"],
        "seed": int(seed),
        "chunk_size": chunk,
        "rollout_seconds": seconds,
        "image_size": list(image.size),
    }
