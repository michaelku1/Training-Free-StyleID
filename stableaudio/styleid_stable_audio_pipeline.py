"""
StyleID-enhanced Stable Audio Open v1.0 inference pipeline for audio-based style transfer.
This implementation adapts StyleID techniques to work with Stable Audio Open's DiT-based architecture.
"""
from __future__ import annotations

import dataclasses
import functools
import inspect
import typing as T
import copy
import pickle
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
# from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils import logging
# from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from transformers import T5Tokenizer, T5EncoderModel
from diffusers import DDPMScheduler

# let python resolve relative import instead of global path
from .test_vae_model_hf_v2 import get_vae_from_stable_audio_open_1_0

device = "cuda" if torch.cuda.is_available() else "cpu"
HF_ACCESS_TOKEN = open("hf_access_token", "r").read()

# NOTE correct flow for using huggingface models
try:
    # Login to Hugging Face (uncomment and run once to authenticate)
    from huggingface_hub import login
    login(HF_ACCESS_TOKEN)  # This will prompt for your token
    print("successfully logged in!")
except Exception as e:
    print(f"Error logging in: {e}")
    print("Make sure you have logged in to Hugging Face with: huggingface-cli login")

# Real audio libraries
try:
    import k_diffusion as K
    K_DIFFUSION_AVAILABLE = True
except ImportError:
    print("Warning: k-diffusion not available. Install with: pip install k-diffusion")
    K_DIFFUSION_AVAILABLE = False

try:
    import encodec
    ENCODEC_AVAILABLE = True
except ImportError:
    print("Warning: encodec not available. Install with: pip install encodec")
    ENCODEC_AVAILABLE = False

# Stable Audio Open v1.0 specific imports
try:
    from stable_audio_tools import get_pretrained_model, create_model_from_config
    STABLE_AUDIO_TOOLS_AVAILABLE = True
    print("Using stable-audio-tools for audio processing")

except Exception as e:
    raise Exception(f"Import error: {e}")



# try:
#     from stable_audio_open import (
#         # StableAudioOpenPipeline,
#         AudioVAE,
#         DiffusionTransformer,
#         AudioScheduler,
#         load_pipeline
#     )
#     STABLE_AUDIO_OPEN_AVAILABLE = True
# except ImportError:
#     print("Warning: stable_audio_open not available. Using stable-audio-tools as alternative.")
#     STABLE_AUDIO_OPEN_AVAILABLE = False
    
# Use stable-audio-tools as the primary alternative
if STABLE_AUDIO_TOOLS_AVAILABLE:
    # Create wrapper classes using stable-audio-tools
    class DiffusionTransformer(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.dtype = torch.float32
            # Use stable-audio-tools model
            try:
                model = get_pretrained_model("stabilityai/stable-audio-open-1.0")
                # breakpoint()
                # NOTE get the model from the wrapper
                dit_wrapper = model[0].model # first argument is the model config, second is metadata
                diffusion_transformer = dit_wrapper.model 
                self.model = diffusion_transformer
                self.model.eval()

            except Exception as e:
                raise Exception(f"Could not load stable-audio-open model: {e}")
            
        def forward(self, x, t):
            # Use stable-audio-tools model
            # return type('obj', (object,), {'sample': self.model(x, t)})()
            return self.model(x, t)

    

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def adain(content_feat, style_feat):
    """
    Adaptive Instance Normalization (AdaIN) for style transfer.
    Transfers the style statistics from style_feat to content_feat.
    """
    content_mean = content_feat.mean(dim=[0, 2, 3], keepdim=True)
    content_std = content_feat.std(dim=[0, 2, 3], keepdim=True)
    style_mean = style_feat.mean(dim=[0, 2, 3], keepdim=True)
    style_std = style_feat.std(dim=[0, 2, 3], keepdim=True)
    output = ((content_feat - content_mean) / content_std) * style_std + style_mean
    return output


def feat_merge(content_feats, style_feats, start_step=0, gamma=0.75, T=1.5):
    """
    Merge content and style features for StyleID injection.
    Adapted to match the diffusers implementation logic.
    
    Args:
        content_feats: Content feature maps from DDIM inversion
        style_feats: Style feature maps from DDIM inversion
        start_step: Starting step for feature injection
        gamma: Query preservation parameter (0-1)
        T: Temperature scaling parameter for attention maps
    """
    # Create merged features structure like in run_styleid_diffusers
    merged_features = {}
    
    # For each layer, merge content and style features
    for layer_name in style_feats.keys():
        merged_features[layer_name] = {}
        
        # For each timestep
        for timestep in style_feats[layer_name].keys():
            if timestep >= start_step:  # Only inject after start_step
                # Get content and style features for this timestep
                content_q, content_k, content_v = content_feats[layer_name][timestep]
                style_q, style_k, style_v = style_feats[layer_name][timestep]
                
                # Style injection: preserve content queries, inject style keys and values
                # This matches the diffusers implementation exactly
                merged_q = content_q * gamma + style_q * (1 - gamma)
                merged_k = style_k
                merged_v = style_v
                
                merged_features[layer_name][timestep] = (merged_q, merged_k, merged_v)
            else:
                # Use content features before start_step
                merged_features[layer_name][timestep] = content_feats[layer_name][timestep]
    
    return merged_features


class StyleIDStableAudioOpenPipeline(DiffusionPipeline):
    """
    StyleID-enhanced Stable Audio Open v1.0 pipeline for audio-based style transfer.
    
    This pipeline extends the original Stable Audio Open v1.0 pipeline with StyleID techniques:
    1. KV style feature injection from style audio into DiT MHSA layers
    2. Query preservation from content audio  
    3. Temperature scaling of attention maps
    4. AdaIN initialization of latents
    
    Adapted to match the diffusers implementation more closely.
    """

    def __init__(
        self,
        audio_vae,
        text_encoder,
        tokenizer,
        diffusion_transformer,
        scheduler,
        # feature_extractor,
    ):
        super().__init__()

        # BUG 
        self.register_modules(
            audio_vae=audio_vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            diffusion_transformer=diffusion_transformer,
            scheduler=scheduler,
            # feature_extractor=feature_extractor,
        )
        
        # StyleID parameters - matching diffusers implementation
        self.gamma = 0.75  # Query preservation parameter
        self.T = 1.5      # Temperature scaling parameter (tau in diffusers)
        self.attn_layers = [0, 1]  # DiT attention layers for injection (adjusted to match actual model)
        self.start_step = 49  # Starting step for feature injection
        
        # Feature storage - matching diffusers implementation
        self.attn_features = {}  # where to save key value (attention block feature)
        self.attn_features_modify = {}  # where to save key value to modify (attention block feature)
        
        # Triggers for feature extraction and injection (like in run_styleid_diffusers)
        self.trigger_get_qkv = False  # If True, save attention features
        self.trigger_modify_qkv = False  # If True, inject modified features
        
        # Current timestep tracking
        self.cur_t = None

    # NOTE override to(device) method to move modules individually to device as diffusers pipeline is not a nn.Module
    def to(self, device):
        for name, module in self.components.items():
            if hasattr(module, "to"):
                self.components[name] = module.to(device)
        return self

    def setup_feature_extraction(self):
        """Setup hooks for extracting attention features during DDIM inversion.
        Adapted to match the diffusers implementation exactly."""
        # Get DiT layers for attention feature extraction
        dit_layers = self._get_dit_layers(self.diffusion_transformer)
        
        # Initialize feature storage
        for i in self.attn_layers:
            if i < len(dit_layers):
                layer_name = f"layer{i}_attn"
                self.attn_features[layer_name] = {}
                self.attn_features_modify[layer_name] = {}
        
        # Register hooks for feature extraction - matching diffusers implementation
        for i in self.attn_layers:
            if i < len(dit_layers):
                layer_name = f"layer{i}_attn"
                # Register hook on DiT attention modules like in diffusers
                if hasattr(dit_layers[i], 'attn'):
                    dit_layers[i].attn.register_forward_hook(
                        self._get_query_key_value(layer_name)
                    )
                    print(f"Hook registered for DiT attention layer {i}: {layer_name}")
                else:
                    # Fallback to direct hook if attn not available
                    dit_layers[i].register_forward_hook(
                        self._get_query_key_value(layer_name)
                    )
                    print(f"Fallback hook registered for DiT attention layer {i}: {layer_name}")
            else:
                print(f"Warning: DiT attention layer {i} not found. Available layers: 0-{len(dit_layers)-1}")
        
        print(f"Target DiT attention layers for injection: {self.attn_layers}")
        print(f"Total DiT layers found: {len(dit_layers)}")
    
    def cleanup_attention_hooks(self):
        """Remove attention hooks to free memory."""
        if hasattr(self, 'attention_hooks'):
            for hook in self.attention_hooks:
                hook.remove()
            self.attention_hooks.clear()
            print("Attention hooks cleaned up")

    def _register_attention_hooks(self):
        """Register hooks to extract attention features from DiT.
        Adapted to match the diffusers implementation."""
        self.attention_hooks = []  # Store hooks for cleanup
        
        # Get DiT layers like in diffusers implementation
        dit_layers = self._get_dit_layers(self.diffusion_transformer)
        
        # Register hooks on specified attention layers - matching diffusers implementation
        for i in self.attn_layers:
            if i < len(dit_layers):
                layer_name = f"layer{i}_attn"
                
                # Register extraction hook on DiT attention modules like in diffusers
                if hasattr(dit_layers[i], 'attn'):
                    hook = dit_layers[i].attn.register_forward_hook(
                        self._get_query_key_value(layer_name)
                    )
                    self.attention_hooks.append(hook)
                    
                    # Also register modification hook
                    modify_hook = dit_layers[i].attn.register_forward_hook(
                        self._modify_self_attn_qkv(layer_name)
                    )
                    self.attention_hooks.append(modify_hook)
                else:
                    # Fallback to direct hook
                    hook = dit_layers[i].register_forward_hook(
                        self._get_query_key_value(layer_name)
                    )
                    self.attention_hooks.append(hook)
                    
                    modify_hook = dit_layers[i].register_forward_hook(
                        self._modify_self_attn_qkv(layer_name)
                    )
                    self.attention_hooks.append(modify_hook)
                
                print(f"Hook registered for DiT attention layer {i}: {layer_name}")
            else:
                print(f"Warning: DiT attention layer {i} not found. Available layers: 0-{len(dit_layers)-1}")
        
        print(f"Target DiT attention layers for injection: {self.attn_layers}")
        print(f"Total DiT layers found: {len(dit_layers)}")
    
    def _get_query_key_value(self, name):
        """Hook function to extract Q, K, V from attention modules.
        Adapted to match the diffusers implementation exactly."""
        def hook(model, input, output):
            if self.trigger_get_qkv:
                # Use the attention_op function like in diffusers implementation
                _, query, key, value, _ = self._attention_op(model, input[0])
                
                self.attn_features[name][int(self.cur_t)] = (query.detach(), key.detach(), value.detach())
            
            # Always return the output
            return output
        return hook
    
    def _modify_self_attn_qkv(self, name):
        """Hook function to inject modified Q, K, V into attention modules.
        Adapted to match the diffusers implementation exactly."""
        def hook(model, input, output):
            if self.trigger_modify_qkv:
                # Get current features
                _, q_cs, k_cs, v_cs, _ = self._attention_op(model, input[0])
                
                # Get stored features for modification
                if name in self.attn_features_modify and int(self.cur_t) in self.attn_features_modify[name]:
                    q_c, k_s, v_s = self.attn_features_modify[name][int(self.cur_t)]
                    
                    # Handle batch size mismatch due to classifier-free guidance
                    # The current batch might be larger due to CFG (uncond + cond)
                    current_batch_size = q_cs.shape[0]
                    stored_batch_size = q_c.shape[0]
                    
                    # in case of classifier-free guidance
                    if current_batch_size != stored_batch_size:
                        # Repeat stored features to match current batch size
                        repeat_factor = current_batch_size // stored_batch_size
                        q_c = q_c.repeat(repeat_factor, 1, 1)
                        k_s = k_s.repeat(repeat_factor, 1, 1)
                        v_s = v_s.repeat(repeat_factor, 1, 1)
                    
                    # Style injection - matching diffusers implementation exactly
                    q_hat_cs = q_c * self.gamma + q_cs * (1 - self.gamma)
                    k_cs, v_cs = k_s, v_s
                    
                    # Replace using attention_op like in diffusers
                    _, _, _, _, modified_output = self._attention_op(
                        model, input[0], 
                        key=k_cs, value=v_cs, query=q_hat_cs, 
                        temperature=self.T
                    )
                    
                    return modified_output
            
            # Always return the output if no modification
            return output
        return hook
    
    def _get_dit_layers(self, diffusion_transformer):
        """
        Get DiT layers for attention feature extraction.
        Adapted to match the diffusers implementation exactly.
        Returns list of DiT transformer blocks.
        """
        # For DiT, we need to find the transformer blocks that contain attention
        # This is adapted from the UNet layer finding logic in diffusers
        dit_layers = []
        
        # Navigate through DiT structure to find attention layers
        # DiT typically has a structure like: transformer_blocks -> attention modules
        if hasattr(diffusion_transformer, 'transformer_blocks'):
            for i, block in enumerate(diffusion_transformer.transformer_blocks):
                if hasattr(block, 'attn'):
                    dit_layers.append(block)
        
        # Fallback: if transformer_blocks not found, try other common structures
        elif hasattr(diffusion_transformer, 'blocks'):
            for i, block in enumerate(diffusion_transformer.blocks):
                if hasattr(block, 'attn'):
                    dit_layers.append(block)
        
        # Fallback: if no clear structure, assume the model itself has attention
        else:
            if hasattr(diffusion_transformer, 'attn'):
                dit_layers.append(diffusion_transformer)


        return dit_layers
    
    def _attention_op(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, query=None, key=None, value=None, attention_probs=None, temperature=1.0):
        """
        Attention operation to get query, key, value and attention map from the DiT.
        Adapted to match the diffusers implementation exactly.
        """
        residual = hidden_states
        
        if hasattr(attn, 'spatial_norm') and attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, None)  # temb not available here

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        
        if hasattr(attn, 'prepare_attention_mask'):
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if hasattr(attn, 'group_norm') and attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if query is None:
            query = attn.to_q(hidden_states)
            query = attn.head_to_batch_dim(query)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif hasattr(attn, 'norm_cross') and attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if key is None:
            key = attn.to_k(encoder_hidden_states)
            key = attn.head_to_batch_dim(key)
        if value is None:
            value = attn.to_v(encoder_hidden_states)
            value = attn.head_to_batch_dim(value)

        # Ensure key and value have the same batch size as query
        if key.shape[0] != query.shape[0]:
            key, value = key[:query.shape[0]], value[:query.shape[0]]

        # apply temperature scaling
        query = query * temperature

        if attention_probs is None:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)

        batch_heads, img_len, txt_len = attention_probs.shape
        
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if hasattr(attn, 'residual_connection') and attn.residual_connection:
            hidden_states = hidden_states + residual

        if hasattr(attn, 'rescale_output_factor'):
            hidden_states = hidden_states / attn.rescale_output_factor
        
        return attention_probs, query, key, value, hidden_states
    
    # NOTE DDIM inversion - adapted to match diffusers implementation
    def extract_features_ddim(self, audio, guidance_scale=7.5, num_steps=50):
        """
        Extract features using DDIM inversion with memory optimization.
        Adapted to match the diffusers implementation.
        
        Args:
            audio: Input audio (content or style)
            num_steps: Number of DDIM inversion steps
            
        Returns:
            Tuple of (latent, features)
        """
        # Setup timestep mapping
        self.scheduler.set_timesteps(num_steps)
        
        # Ensure timesteps are available
        if not hasattr(self.scheduler, 'timesteps') or self.scheduler.timesteps is None:
            raise ValueError("Scheduler timesteps not properly initialized")
            
        # Convert to numpy array if it's a tensor
        timesteps = self.scheduler.timesteps.cpu().numpy() if torch.is_tensor(self.scheduler.timesteps) else self.scheduler.timesteps
        time_range = np.flip(timesteps)
        self.idx_time_dict = {}
        self.time_idx_dict = {}
        for i, t in enumerate(time_range):
            self.idx_time_dict[t] = i
            self.time_idx_dict[i] = t

        # Encode audio to latent using AudioVAE
        if isinstance(audio, torch.Tensor):
            device = audio.device  # Use the same device as input audio
            audio_tensor = audio.to(device=device, dtype=self.audio_vae.dtype)
        else:
            # Convert audio to tensor format expected by AudioVAE
            audio_tensor = self._preprocess_audio(audio)
        
        # encode spectrogram to latent
        init_latent_dist = self.audio_vae.encode(audio_tensor).latent_dist
        init_latents = init_latent_dist.sample()
        # AudioVAE might have different scaling factor than image VAE
        init_latents = 0.18215 * init_latents  # scale latents

        # DDIM inversion with feature extraction - matching diffusers implementation
        latents = init_latents.clone()
        
        # Reversed timesteps like in diffusers implementation
        timesteps = list(reversed(self.scheduler.timesteps))
        num_inference_steps = len(self.scheduler.timesteps)
        cur_latent = latents.clone()

        with torch.no_grad():
            for i in range(0, num_inference_steps):
                t = timesteps[i]
                
                # Set current timestep for attention hooks
                self.cur_t = t.item() if hasattr(t, 'item') else t
                
                # Predict the noise residual
                noise_pred = self.diffusion_transformer(cur_latent, t.to(cur_latent.device)).sample
                
                # For text condition on stable diffusion
                if noise_pred.shape[0] == 2:
                    # perform guidance
                    noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    # NOTE: for sampling, we just need one latent
                    cur_latent, _ = cur_latent.chunk(2)

                # DDIM inversion formula - matching diffusers implementation
                current_t = max(0, t.item() - (1000//num_inference_steps))
                next_t = t
                alpha_t = self.scheduler.alphas_cumprod[current_t]
                alpha_t_next = self.scheduler.alphas_cumprod[next_t]
                
                # Inverted update step (re-arranging the update step to get x(t) as a function of x(t-1))
                cur_latent = (cur_latent - (1-alpha_t).sqrt()*noise_pred)*(alpha_t_next.sqrt()/alpha_t.sqrt()) + (1-alpha_t_next).sqrt()*noise_pred
                
                # Clear intermediate tensors to save memory
                del noise_pred
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return cur_latent, None  # Return final latent and None for features (handled by hooks)

    def _preprocess_audio(self, audio):
        """
        Preprocess audio for AudioVAE encoding.
        This method should be adapted based on the specific AudioVAE requirements.
        """
        # Placeholder implementation - adapt based on actual AudioVAE requirements
        if isinstance(audio, str):
            # Load audio file
            import soundfile as sf
            audio_data, sample_rate = sf.read(audio)
            audio_tensor = torch.from_numpy(audio_data).float()
        elif isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio
            
        # Ensure correct shape for AudioVAE (batch, channels, time)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
            
        # Use the device of the pipeline components
        device = next(self.parameters()).device if hasattr(self, 'parameters') else torch.device('cpu')
        return audio_tensor.to(device=device, dtype=self.audio_vae.dtype)

    def _encode_text(self, prompt: str):
        """Encode text prompt using the text encoder."""
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        device = torch.device('cpu')  # Default to CPU for testing
        text_embeddings = self.text_encoder(inputs.input_ids.to(device))[0]
        return text_embeddings

    def _decode_audio(self, latents):
        """Decode latents to audio using AudioVAE."""
        latents = 1.0 / 0.18215 * latents  # Adjust scaling factor if needed
        audio = self.audio_vae.decode(latents).sample
        return audio

    def _save_audio(self, audio, output_path):
        """Save audio to file."""
        import soundfile as sf
        # Convert to numpy and save
        audio_np = audio.cpu().numpy()
        sf.write(output_path, audio_np, samplerate=44100)  # Adjust sample rate as needed

    @torch.no_grad()
    def styleid_generate(
        self,
        prompt: str,
        init_latents: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,  # Kept for compatibility but not used
        eta: T.Optional[float] = 0.0,
        start_step: int = 49,
        device: str = "cuda",
    ):
        """
        DDIM sampling (reverse process)

        StyleID-enhanced audio generation with attention feature injection.
        Adapted to match the diffusers implementation.
        
        Note: Stable Audio Open v1.0 doesn't support text conditioning, so text_embeddings,
        guidance_scale, and negative_prompt are kept for compatibility but not used in the actual generation.
        """

        pred_images = []
        pred_latents = []

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # NOTE: text condition (conditional + unconditional text embeddings) on stable diffusion
        text_condition_embedding = self.get_text_condition(prompt, device=device)

        # Prepare extra kwargs for scheduler
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        timesteps = self.scheduler.timesteps.to(device)

        # NOTE DDIM sampling (reverse process)
        for i, t in enumerate(self.progress_bar(timesteps)):
            # Set current timestep for attention hooks
            self.cur_t = t.item() if hasattr(t, 'item') else t
            
            # Skip injection before start_step
            if i < start_step:
                self.trigger_modify_qkv = False
            else:
                self.trigger_modify_qkv = True

            # NOTE Predict noise with StyleID injection
            noise_pred = self.diffusion_transformer(
                latent_model_input, 
                t,
                text_condition_embedding,
            ).sample

            # NOTE perform guidance on conditional and unconditional unet outputs
            unconditional_output, conditional_output = noise_pred.chunk(2)
            noise_pred = conditional_output + guidance_scale * (conditional_output - unconditional_output)


            # NOTE sampling: Compute previous noisy sample
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # Standard DDIM step: move from t to t-1 (denoising)
            prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample                # coef * P_t(e_t(x_t)) + D_t(e_t(x_t))
            pred_original_sample = scheduler.step(noisy_residual, t, input).pred_original_sample    # D_t(e_t(x_t))
            # update sample
            input = prev_noisy_sample
            
            # save latents
            pred_latents.append(pred_original_sample)
            # save images (decoded latents)
            pred_images.append(decode_latent(pred_original_sample, **decode_kwargs))

        return dict(latents=latents)
    
    def get_text_condition(self, text, device="cuda"):
        # NOTE if nothing is passed, use empty prompt
        if text is None:
            uncond_input = self.tokenizer(
                [""], padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt"
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0].to(device)
            return {'encoder_hidden_states': uncond_embeddings}
        
        text_embeddings, uncond_embeddings = self.get_text_embedding(text, self.text_encoder, self.tokenizer)
        text_cond = [text_embeddings, uncond_embeddings]
        denoise_kwargs = {
            'encoder_hidden_states': torch.cat(text_cond)
        }
        return denoise_kwargs
    
    def get_text_embedding(self, text, text_encoder, tokenizer, device="cuda"):
        """
        Get both text and uncond text embeddings for text conditioning.
        
        Args:
            text: Text prompt
            text_encoder: Text encoder
            tokenizer: Tokenizer
            device: Device to run the model on
        """
        
        # TODO currently, hard-coding for stable diffusion
        with torch.no_grad():

            prompt = [text]
            batch_size = len(prompt)
            text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

            text_embeddings = text_encoder(text_input.input_ids.to(device))[0].to(device)
            max_length = text_input.input_ids.shape[-1]
            # print(max_length, text_input.input_ids)
            uncond_input = tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0].to(device)
        
        return text_embeddings, uncond_embeddings

    def inject_mhsa_style_features(self, content_features, style_features, gamma=0.75, T=1.5):
        """
        Inject style features into MHSA layers using StyleID technique.
        
        Args:
            content_features: Content MHSA features (Q, K, V)
            style_features: Style MHSA features (Q, K, V)
            gamma: Query preservation parameter
            T: Temperature scaling parameter
            
        Returns:
            Injected MHSA features
        """
        if not content_features or not style_features:
            return content_features
        
        injected_features = {}
        
        # Apply StyleID injection to Q, K, V
        for key in ['q', 'k', 'v']:
            if key in content_features and key in style_features:
                content_feat = content_features[key]
                style_feat = style_features[key]
                
                if key == 'q':
                    # Query preservation: keep content query
                    injected_features[key] = content_feat
                else:
                    # Inject style key and value with temperature scaling
                    injected_features[key] = style_feat / T
        
        return injected_features

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: str,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        local_files_only: bool = False,
        low_cpu_mem_usage: bool = False,
        cache_dir: T.Optional[str] = None,
    ) -> StyleIDStableAudioOpenPipeline:
        """
        Load a StyleID Stable Audio Open v1.0 pipeline from a checkpoint.
        
        Args:
            checkpoint: Path to the model checkpoint
            device: Device to load the model on
            dtype: Data type for model weights
            local_files_only: Don't download, only use local files
            low_cpu_mem_usage: Attempt to use less memory on CPU
        """
        device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")

        if device.type == "cpu" or device.type == "mps":
            print(f"WARNING: Falling back to float32 on {device}, float16 is unsupported")
            dtype = torch.float32

        try:
            # Load Stable Audio Open v1.0 components
            # if STABLE_AUDIO_OPEN_AVAILABLE:
            #     # Load the base Stable Audio Open v1.0 pipeline
            #     base_pipeline = load_pipeline(checkpoint, device=device, dtype=dtype)
                
            #     # Extract components from Stable Audio Open v1.0 pipeline
            #     audio_vae = base_pipeline.audio_vae
            #     text_encoder = base_pipeline.text_encoder
            #     tokenizer = base_pipeline.tokenizer
            #     diffusion_transformer = base_pipeline.diffusion_transformer
            #     scheduler = base_pipeline.scheduler
            #     feature_extractor = base_pipeline.feature_extractor
            # else:

            # Use real audio libraries as alternatives
            # BUG: this wrapper is not working, need to use the vae directly from the stable-audio-tools library
            # audio_vae = AutoencoderKL.from_pretrained("stabilityai/stable-audio-open-1.0", subfolder="vae")

            audio_vae = get_vae_from_stable_audio_open_1_0()
            text_encoder = T5EncoderModel.from_pretrained(checkpoint, subfolder="text_encoder") # ok
            tokenizer = T5Tokenizer.from_pretrained(checkpoint, subfolder="tokenizer") # ok

            breakpoint()
            diffusion_transformer = DiffusionTransformer()
            scheduler = DDPMScheduler.from_pretrained(checkpoint, subfolder="scheduler")
            # feature_extractor = AutoFeatureExtractor.from_pretrained("stabilityai/stable-audio-open-1.0") # ok
            
            # Create the StyleIDStableAudioOpenPipeline
            styleid_pipeline = cls(
                audio_vae=audio_vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                diffusion_transformer=diffusion_transformer,
                scheduler=scheduler,
                # feature_extractor=feature_extractor,
            )


            # Move to device
            styleid_pipeline = styleid_pipeline.to(device)

            return styleid_pipeline
            
        except Exception as e:
            print(f"Error loading Stable Audio Open v1.0 pipeline: {e}")
            raise
        
