from typing import Any, Dict, List, Optional, Union

import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXDDIMScheduler, CogVideoXPipeline
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer

from .utils import prepare_rotary_positional_embeddings
from .model import CogVideoXTransformer3DModel

def load_condition_models(
    model_id: str = "THUDM/CogVideoX-5b",
    text_encoder_dtype: torch.dtype = torch.bfloat16,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
):
    tokenizer = T5Tokenizer.from_pretrained(model_id, subfolder="tokenizer", revision=revision, cache_dir=cache_dir)
    text_encoder = T5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=text_encoder_dtype, revision=revision, cache_dir=cache_dir
    )
    return {"tokenizer": tokenizer, "text_encoder": text_encoder}


def load_latent_models(
    model_id: str = "THUDM/CogVideoX-5b",
    vae_dtype: torch.dtype = torch.bfloat16,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
):
    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_id, subfolder="vae", torch_dtype=vae_dtype, revision=revision, cache_dir=cache_dir
    )
    return {"vae": vae}


def load_diffusion_models(
    model_id: str = "THUDM/CogVideoX-5b",
    transformer_dtype: torch.dtype = torch.bfloat16,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
):
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=transformer_dtype, revision=revision, cache_dir=cache_dir
    )
    scheduler = CogVideoXDDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    return {"transformer": transformer, "scheduler": scheduler}


def initialize_pipeline(
    model_id: str = "THUDM/CogVideoX-5b",
    text_encoder_dtype: torch.dtype = torch.bfloat16,
    transformer_dtype: torch.dtype = torch.bfloat16,
    vae_dtype: torch.dtype = torch.bfloat16,
    tokenizer: Optional[T5Tokenizer] = None,
    text_encoder: Optional[T5EncoderModel] = None,
    transformer: Optional[CogVideoXTransformer3DModel] = None,
    vae: Optional[AutoencoderKLCogVideoX] = None,
    scheduler: Optional[CogVideoXDDIMScheduler] = None,
    device: Optional[torch.device] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    enable_slicing: bool = False,
    enable_tiling: bool = False,
    enable_model_cpu_offload: bool = False,
    is_training: bool = False,
    **kwargs,
) -> CogVideoXPipeline:
    component_name_pairs = [
        ("tokenizer", tokenizer),
        ("text_encoder", text_encoder),
        ("transformer", transformer),
        ("vae", vae),
        ("scheduler", scheduler),
    ]
    components = {}
    for name, component in component_name_pairs:
        if component is not None:
            components[name] = component

    pipe = CogVideoXPipeline.from_pretrained(model_id, **components, revision=revision, cache_dir=cache_dir)
    pipe.text_encoder = pipe.text_encoder.to(dtype=text_encoder_dtype)
    pipe.vae = pipe.vae.to(dtype=vae_dtype)

    # The transformer should already be in the correct dtype when training, so we don't need to cast it here.
    # If we cast, whilst using fp8 layerwise upcasting hooks, it will lead to an error in the training during
    # DDP optimizer step.
    if not is_training:
        pipe.transformer = pipe.transformer.to(dtype=transformer_dtype)

    if enable_slicing:
        pipe.vae.enable_slicing()
    if enable_tiling:
        pipe.vae.enable_tiling()

    if enable_model_cpu_offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device=device)

    return pipe


def prepare_conditions(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    max_sequence_length: int = 226,  # TODO: this should be configurable
    **kwargs,
):
    device = device or text_encoder.device
    dtype = dtype or text_encoder.dtype
    return _get_t5_prompt_embeds(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt=prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
    )


def prepare_latents(
    vae: AutoencoderKLCogVideoX,
    image_or_video: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
    precompute: bool = False,
    **kwargs,
) -> torch.Tensor:
    device = device or vae.device
    dtype = dtype or vae.dtype

    if image_or_video.ndim == 4:
        image_or_video = image_or_video.unsqueeze(2)
    assert image_or_video.ndim == 5, f"Expected 5D tensor, got {image_or_video.ndim}D tensor"

    image_or_video = image_or_video.to(device=device, dtype=vae.dtype)
    image_or_video = image_or_video.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
    if not precompute:
        latents = vae.encode(image_or_video).latent_dist.sample(generator=generator)
        if not vae.config.invert_scale_latents:
            latents = latents * vae.config.scaling_factor
        # For training Cog 1.5, we don't need to handle the scaling factor here.
        # The CogVideoX team forgot to multiply here, so we should not do it too. Invert scale latents
        # is probably only needed for image-to-video training.
        # TODO(aryan): investigate this
        # else:
        #     latents = 1 / vae.config.scaling_factor * latents
        latents = latents.to(dtype=dtype)
        return {"latents": latents}
    else:
        # handle vae scaling in the `train()` method directly.
        if vae.use_slicing and image_or_video.shape[0] > 1:
            encoded_slices = [vae._encode(x_slice) for x_slice in image_or_video.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = vae._encode(image_or_video)
        return {"latents": h}


def post_latent_preparation(
    vae_config: Dict[str, Any], latents: torch.Tensor, patch_size_t: Optional[int] = None, **kwargs
) -> torch.Tensor:
    if not vae_config.invert_scale_latents:
        latents = latents * vae_config.scaling_factor
    # For training Cog 1.5, we don't need to handle the scaling factor here.
    # The CogVideoX team forgot to multiply here, so we should not do it too. Invert scale latents
    # is probably only needed for image-to-video training.
    # TODO(aryan): investigate this
    # else:
    #     latents = 1 / vae_config.scaling_factor * latents
    latents = _pad_frames(latents, patch_size_t)
    latents = latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
    return {"latents": latents}


def collate_fn_t2v(batch: List[List[Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    return {
        "prompts": [x["prompt"] for x in batch[0]],
        "videos": torch.stack([x["video"] for x in batch[0]]),
    }


def calculate_noisy_latents(
    scheduler: CogVideoXDDIMScheduler,
    noise: torch.Tensor,
    latents: torch.Tensor,
    timesteps: torch.LongTensor,
) -> torch.Tensor:
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    return noisy_latents


def forward_pass(
    transformer: CogVideoXTransformer3DModel,
    scheduler: CogVideoXDDIMScheduler,
    prompt_embeds: torch.Tensor,
    latents: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.LongTensor,
    ofs_emb: Optional[torch.Tensor] = None,
    return_hidden_states: Optional[List[int]] = None,
    apply_target_noise_only: bool = False,
    **kwargs,
) -> torch.Tensor:
    # Just hardcode for now. In Diffusers, we will refactor such that RoPE would be handled within the model itself.
    VAE_SPATIAL_SCALE_FACTOR = 8
    transformer_config = transformer.module.config if hasattr(transformer, "module") else transformer.config
    batch_size, num_frames, num_channels, height, width = noisy_latents.shape
    rope_base_height = transformer_config.sample_height * VAE_SPATIAL_SCALE_FACTOR
    rope_base_width = transformer_config.sample_width * VAE_SPATIAL_SCALE_FACTOR

    image_rotary_emb = (
        prepare_rotary_positional_embeddings(
            height=height * VAE_SPATIAL_SCALE_FACTOR,
            width=width * VAE_SPATIAL_SCALE_FACTOR,
            num_frames=num_frames,
            vae_scale_factor_spatial=VAE_SPATIAL_SCALE_FACTOR,
            patch_size=transformer_config.patch_size,
            patch_size_t=transformer_config.patch_size_t if hasattr(transformer_config, "patch_size_t") else None,
            attention_head_dim=transformer_config.attention_head_dim,
            device=transformer.device,
            base_height=rope_base_height,
            base_width=rope_base_width,
        )
        if transformer_config.use_rotary_positional_embeddings
        else None
    )
    ofs_emb = None if transformer_config.ofs_embed_dim is None else latents.new_full((batch_size,), fill_value=2.0)

    if return_hidden_states is not None:
        if return_hidden_states == "all":
            return_hidden_states = list(range(1, transformer.num_layers + 1))
        else:
            return_hidden_states = [int(layer) for layer in return_hidden_states.split(",")]


    output = transformer(
        hidden_states=noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        ofs=ofs_emb,
        image_rotary_emb=image_rotary_emb,
        return_dict=False,
        return_hidden_states=return_hidden_states,
        apply_target_noise_only=apply_target_noise_only,
    )
    if return_hidden_states is not None:
        velocity, hidden_states_list = output
    else:
        velocity = output[0]

    # For CogVideoX, the transformer predicts the velocity. The denoised output is calculated by applying the same
    # code paths as scheduler.get_velocity(), which can be confusing to understand.
    denoised_latents = scheduler.get_velocity(velocity, noisy_latents, timesteps)
    if return_hidden_states is not None:
        return {"latents": denoised_latents, "hidden_states": hidden_states_list}
    else:
        return {"latents": denoised_latents}

import math
import types
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler

def validation(
    pipeline: CogVideoXPipeline,
    prompt: str,
    image: Optional[Image.Image] = None,
    video: Optional[List[Image.Image]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_frames: Optional[int] = None,
    num_videos_per_prompt: int = 1,
    generator: Optional[torch.Generator] = None,
    apply_target_noise_only = None,
    enable_model_cpu_offload = False,
    **kwargs,
):
    init_latents = process_video(pipeline, video, pipeline.dtype, generator, height, width, apply_target_noise_only)
    pipeline.custom_call = types.MethodType(custom_call, pipeline)

    generation_kwargs = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "latents": init_latents,
        "num_frames": num_frames,
        "num_videos_per_prompt": num_videos_per_prompt,
        "generator": generator,
        "return_dict": True,
        "output_type": "pil",
        "apply_target_noise_only": apply_target_noise_only,
    }
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
    output = pipeline.custom_call(**generation_kwargs).frames[0]
    return [("video", output)]

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

@torch.no_grad()
def process_video(pipe, video, dtype, generator, height, width, apply_target_noise_only):
    if pipe.device != "cuda":
        generator = None
    if apply_target_noise_only == None:
        return None
    from diffusers.pipelines.cogvideo.pipeline_cogvideox_video2video import retrieve_latents
    from diffusers.utils.torch_utils import randn_tensor
    video = pipe.video_processor.preprocess_video(video, height=height, width=width)
    video = video.to("cuda", dtype=dtype)
    
    video_latents = retrieve_latents(pipe.vae.encode(video))
    init_latents = video_latents.to(dtype).permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
    init_latents = pipe.vae_scaling_factor_image * init_latents
    init_latents = init_latents.to(pipe.device)

    noise = randn_tensor(init_latents.shape, generator=generator, device=pipe.device, dtype=dtype)
    print(f"[DEBUG] noise applied: {apply_target_noise_only}")
    if apply_target_noise_only == "back":
        init_latents[:, :-1] = noise[:, :-1]
    elif apply_target_noise_only == "front":
        init_latents[:, 1:] = noise[:, 1:]
    elif apply_target_noise_only == "front-long":
        init_latents[:, 6:] = noise[:, 6:]
    elif apply_target_noise_only == "front-last-long":
        init_latents[:, 6:-1] = noise[:, 6:-1]
    elif apply_target_noise_only == "front-last-long-long":
        init_latents[:, 5:-3] = noise[:, 5:-3]
    elif apply_target_noise_only == "front-2":
        init_latents[:, 2:] = noise[:, 2:]
    elif apply_target_noise_only == 'front-4-none':
        init_latents[:, 4:] = noise[:, 4:]
    elif apply_target_noise_only == "front-4-noise-none":
        timesteps = pipe.scheduler.timesteps # torch.Size([1000]), torch.float32, 999~0
        scheduler = pipe.scheduler
        n_timesteps = timesteps.shape[0]
        #t_100 = timesteps[0]
        t_25 = timesteps[int(n_timesteps * (1 - 0.25))]
        t_50 = timesteps[int(n_timesteps * (1 - 0.5))]
        t_75 = timesteps[int(n_timesteps * (1 - 0.75))]
        print(f"[DEBUG] applied noise mode : {apply_target_noise_only}")
        #init_latents[:, :, 0] = scheduler.add_noise(init_latents[:, :, 0], noise[:, :, 0], torch.tensor([t_100]))
        init_latents[:, 1] = scheduler.add_noise(init_latents[:, 1], noise[:, 1], torch.tensor([t_25]))
        init_latents[:, 2] = scheduler.add_noise(init_latents[:, 2], noise[:, 2], torch.tensor([t_50]))
        init_latents[:, 3] = scheduler.add_noise(init_latents[:, 3], noise[:, 3], torch.tensor([t_75]))
        init_latents[:, 4:] = noise[:, 4:]
    elif apply_target_noise_only == "front-7-noise-none":
        timesteps = pipe.scheduler.timesteps # torch.Size([1000]), torch.float32, 999~0
        scheduler = pipe.scheduler
        n_timesteps = timesteps.shape[0]
        #t_100 = timesteps[0]
        t_25 = timesteps[int(n_timesteps * (1 - 0.25))]
        t_50 = timesteps[int(n_timesteps * (1 - 0.5))]
        t_75 = timesteps[int(n_timesteps * (1 - 0.75))]
        print(f"[DEBUG] applied noise mode : {apply_target_noise_only}")
        #init_latents[:, :, 0] = scheduler.add_noise(init_latents[:, :, 0], noise[:, :, 0], torch.tensor([t_100]))
        init_latents[:, 4] = scheduler.add_noise(init_latents[:, 4], noise[:, 4], torch.tensor([t_25]))
        init_latents[:, 5] = scheduler.add_noise(init_latents[:, 5], noise[:, 5], torch.tensor([t_50]))
        init_latents[:, 6] = scheduler.add_noise(init_latents[:, 6], noise[:, 6], torch.tensor([t_75]))
        init_latents[:, 7:] = noise[:, 7:]
    elif apply_target_noise_only == "none":
        init_latents = noise
    else:
        raise ValueError(f"apply_target_noise_only must be either 'back' or 'front', but got {apply_target_noise_only}")
    init_latents = init_latents.to(pipe.device)
    return init_latents

@torch.no_grad()
def retrieve_video(pipe, init_latents,):
    init_latents = init_latents.to("cuda")
    video = pipe.decode_latents(init_latents)
    video = pipe.video_processor.postprocess_video(video=video, output_type="pil")[0]
    return video

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

@torch.no_grad()
def custom_call(
    self,
    prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_frames: Optional[int] = None,
    num_inference_steps: int = 50,
    timesteps: Optional[List[int]] = None,
    guidance_scale: float = 6,
    use_dynamic_cfg: bool = False,
    num_videos_per_prompt: int = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: str = "pil",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 226,
    apply_target_noise_only: bool = False,
) -> Union[CogVideoXPipelineOutput, Tuple]:
    print("<Custom Call>")
    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
    width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
    num_frames = num_frames or self.transformer.config.sample_frames

    num_videos_per_prompt = 1

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds,
        negative_prompt_embeds,
    )
    self._guidance_scale = guidance_scale
    self._attention_kwargs = attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    # 2. Default call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt,
        negative_prompt,
        do_classifier_free_guidance,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        max_sequence_length=max_sequence_length,
        device=device,
    )
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
    self._num_timesteps = len(timesteps)

    # 5. Prepare latents
    latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

    # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
    patch_size_t = self.transformer.config.patch_size_t
    additional_frames = 0
    if patch_size_t is not None and latent_frames % patch_size_t != 0:
        additional_frames = patch_size_t - latent_frames % patch_size_t
        num_frames += additional_frames * self.vae_scale_factor_temporal

    latent_channels = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_videos_per_prompt,
        latent_channels,
        num_frames,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Create rotary embeds if required
    image_rotary_emb = (
        self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
        if self.transformer.config.use_rotary_positional_embeddings
        else None
    )

    # 8. Denoising loop
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

    @torch.no_grad()
    def retrieve_video(init_latents,):
        init_latents = init_latents.to("cuda")
        video = self.decode_latents(init_latents)
        video = self.video_processor.postprocess_video(video=video, output_type="pil")[0]
        return video

    print(f"[pipeline] apply_target_noise_only: {apply_target_noise_only}")
    if apply_target_noise_only == "front-4-noise-none" or apply_target_noise_only == "front-7-noise-none":
        timesteps = self.scheduler.timesteps # torch.Size([1000]), torch.float32, 999~0
        scheduler = self.scheduler
        n_timesteps = timesteps.shape[0]
        #t_100 = timesteps[0]
        t_25 = timesteps[int(n_timesteps * (1 - 0.25))]
        t_50 = timesteps[int(n_timesteps * (1 - 0.5))]
        t_75 = timesteps[int(n_timesteps * (1 - 0.75))]

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        # for DPM-solver++
        old_pred_original_sample = None
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            self._current_timestep = t
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])

            # video = retrieve_video(latents)
            # from diffusers.utils import export_to_video
            # export_to_video(video, f"my_test/video_{i}.mp4")

            # predict noise model_output
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
                return_dict=False,
                apply_target_noise_only=apply_target_noise_only,
            )[0]
            noise_pred = noise_pred.float()

            # perform guidance
            if use_dynamic_cfg:
                self._guidance_scale = 1 + guidance_scale * (
                    (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                )
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                if apply_target_noise_only == "front":
                    noise_pred[:, 0] = 0
                elif apply_target_noise_only == "back":
                    noise_pred[:, -1] = 0
                elif apply_target_noise_only == "front-long":
                    noise_pred[:, :6] = 0
                elif apply_target_noise_only == "front-last-long":
                    noise_pred[:, :6] = 0
                    noise_pred[:, -1] = 0
                elif apply_target_noise_only == "front-last-long-long":
                    noise_pred[:, :5] = 0
                    noise_pred[:, -3:] = 0
                elif apply_target_noise_only == "front-2":
                    noise_pred[:, :2] = 0
                elif apply_target_noise_only == "front-4-none":
                    noise_pred[:, :4] = 0
                elif apply_target_noise_only == "front-4-noise-none":
                    noise_pred[:, 0] = 0
                    if t > t_25:
                        print(f"[DEBUG] not reached t_25")
                        noise_pred[:, 1] = 0
                    if t > t_50:
                        print(f"[DEBUG] not reached t_50")
                        noise_pred[:, 2] = 0
                    if t > t_75:
                        print(f"[DEBUG] not reached t_75")
                        noise_pred[:, 3] = 0
                elif apply_target_noise_only == "front-7-noise-none":
                    noise_pred[:, :4] = 0
                    if t > t_25:
                        print(f"[DEBUG] not reached t_25")
                        noise_pred[:, 4] = 0
                    if t > t_50:
                        print(f"[DEBUG] not reached t_50")
                        noise_pred[:, 5] = 0
                    if t > t_75:
                        print(f"[DEBUG] not reached t_75")
                        noise_pred[:, 6] = 0
                elif apply_target_noise_only == "none":
                    pass
                else:
                    raise NotImplementedError
                

            # compute the previous noisy sample x_t -> x_t-1
            if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            else:
                latents, old_pred_original_sample = self.scheduler.step(
                    noise_pred,
                    old_pred_original_sample,
                    t,
                    timesteps[i - 1] if i > 0 else None,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )
            latents = latents.to(prompt_embeds.dtype)

            # call the callback, if provided
            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

    self._current_timestep = None

    if not output_type == "latent":
        # Discard any padding frames that were added for CogVideoX 1.5
        latents = latents[:, additional_frames:]
        video = self.decode_latents(latents)
        video = self.video_processor.postprocess_video(video=video, output_type=output_type)
    else:
        video = latents

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (video,)

    return CogVideoXPipelineOutput(frames=video)

def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]] = None,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    return {"prompt_embeds": prompt_embeds}


def _pad_frames(latents: torch.Tensor, patch_size_t: int):
    if patch_size_t is None or patch_size_t == 1:
        return latents

    # `latents` should be of the following format: [B, C, F, H, W].
    # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
    latent_num_frames = latents.shape[2]
    additional_frames = patch_size_t - latent_num_frames % patch_size_t

    if additional_frames > 0:
        last_frame = latents[:, :, -1:, :, :]
        padding_frames = last_frame.repeat(1, 1, additional_frames, 1, 1)
        latents = torch.cat([latents, padding_frames], dim=2)

    return latents


# TODO(aryan): refactor into model specs for better re-use
COGVIDEOX_T2V_LORA_CONFIG = {
    "pipeline_cls": CogVideoXPipeline,
    "load_condition_models": load_condition_models,
    "load_latent_models": load_latent_models,
    "load_diffusion_models": load_diffusion_models,
    "initialize_pipeline": initialize_pipeline,
    "prepare_conditions": prepare_conditions,
    "prepare_latents": prepare_latents,
    "post_latent_preparation": post_latent_preparation,
    "collate_fn": collate_fn_t2v,
    "calculate_noisy_latents": calculate_noisy_latents,
    "forward_pass": forward_pass,
    "validation": validation,
}
