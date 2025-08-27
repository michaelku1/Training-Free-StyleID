"""
StyleID-enhanced Riffusion inference pipeline for spectrogram-based style transfer.
This implementation adapts StyleID techniques to work with Riffusion's spectrogram processing.
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
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils import logging
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from einops import rearrange

from riffusion.datatypes import InferenceInput
from riffusion.external.prompt_weighting import get_weighted_text_embeddings
from riffusion.util import torch_util
from riffusion.riffusion_pipeline import RiffusionPipeline

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

# NOTE performing style injection at specified layers - adapted to match diffusers implementation
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


class StyleIDRiffusionPipeline(RiffusionPipeline):
    """
    StyleID-enhanced Riffusion pipeline for spectrogram-based style transfer.
    
    This pipeline extends the original Riffusion pipeline with StyleID techniques:
    1. KV style feature injection from style spectrograms
    2. Query preservation from content spectrograms  
    3. Temperature scaling of attention maps
    4. AdaIN initialization of latents
    
    Adapted to match the diffusers implementation more closely.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: T.Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        attention_op_type: str = '1',
        # feature_extractor: CLIPFeatureExtractor, # NOTE coming from huggingface transformers
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            # feature_extractor=feature_extractor,
        )
        
        # StyleID parameters - matching diffusers implementation
        self.gamma = 0.75  # Query preservation parameter
        self.T = 1.5      # Temperature scaling parameter (tau in diffusers)
        self.attn_layers = [7, 8, 9, 10, 11]  # Attention layers for injection (matching run_styleid_diffusers)
        self.start_step = 49  # Starting step for feature injection
        
        # Feature storage - matching diffusers implementation
        self.attn_features = {}  # where to save key value (attention block feature)
        self.attn_features_modify = {}  # where to save key value to modify (attention block feature)
        
        # Triggers for feature extraction and injection (like in run_styleid_diffusers)
        self.trigger_get_qkv = False  # If True, save attention features
        self.trigger_modify_qkv = False  # If True, inject modified features
        
        # Current timestep tracking
        self.cur_t = None

        # TODO add different attention op types
        self.attention_op_type = attention_op_type

        if attention_op_type == '1':
            self._attention_op = self._attention_op_embeddings
        elif attention_op_type == '2':
            self._attention_op = self._attention_op_output_scores_interp
        elif attention_op_type == '3':
            self._attention_op = self._attention_op_residual_outputs
        else:
            raise ValueError(f"Invalid attention op type: {attention_op_type}")


    def setup_feature_extraction(self):
        """Setup hooks for extracting attention features during DDIM inversion.
        Adapted to match the diffusers implementation exactly."""
        # Get residual and attention blocks like in diffusers implementation
        resnet, attn = self._get_unet_layers(self.unet)
        
        # Initialize feature storage
        for i in self.attn_layers:
            if i < len(attn):
                layer_name = f"layer{i}_attn"
                self.attn_features[layer_name] = {}
                self.attn_features_modify[layer_name] = {}
        
        # Register hooks for feature extraction - matching diffusers implementation
        for i in self.attn_layers:
            if i < len(attn):
                layer_name = f"layer{i}_attn"
                # Register hook on transformer_blocks[0].attn1 like in diffusers
                if hasattr(attn[i], 'transformer_blocks') and len(attn[i].transformer_blocks) > 0:
                    attn[i].transformer_blocks[0].attn1.register_forward_hook(
                        self._get_query_key_value(layer_name)
                    )
                    print(f"Hook registered for attention layer {i}: {layer_name}")
                else:
                    # Fallback to direct hook if transformer_blocks not available
                    attn[i].register_forward_hook(
                        self._get_query_key_value(layer_name)
                    )
                    print(f"Fallback hook registered for attention layer {i}: {layer_name}")
            else:
                print(f"Warning: Attention layer {i} not found. Available layers: 0-{len(attn)-1}")
        
        print(f"Target attention layers for injection: {self.attn_layers}")
        print(f"Total attention layers found: {len(attn)}")
    
    def cleanup_attention_hooks(self):
        """Remove attention hooks to free memory."""
        if hasattr(self, 'attention_hooks'):
            for hook in self.attention_hooks:
                hook.remove()
            self.attention_hooks.clear()
            print("Attention hooks cleaned up")

    def _register_attention_hooks(self):
        """Register hooks to extract attention features from UNet.
        Adapted to match the diffusers implementation."""
        self.attention_hooks = []  # Store hooks for cleanup
        
        # Get UNet layers like in diffusers implementation
        resnet, attn = self._get_unet_layers(self.unet)
        
        # Register hooks on specified attention layers - matching diffusers implementation
        for i in self.attn_layers:
            if i < len(attn):
                layer_name = f"layer{i}_attn"
                
                # Register extraction hook on transformer_blocks[0].attn1 like in diffusers
                if hasattr(attn[i], 'transformer_blocks') and len(attn[i].transformer_blocks) > 0:
                    hook = attn[i].transformer_blocks[0].attn1.register_forward_hook(
                        self._get_query_key_value(layer_name)
                    )
                    self.attention_hooks.append(hook)
                    
                    # Also register modification hook
                    modify_hook = attn[i].transformer_blocks[0].attn1.register_forward_hook(
                        self._modify_self_attn_qkv(layer_name)
                    )
                    self.attention_hooks.append(modify_hook)
                else:
                    # Fallback to direct hook
                    hook = attn[i].register_forward_hook(
                        self._get_query_key_value(layer_name)
                    )
                    self.attention_hooks.append(hook)
                    
                    modify_hook = attn[i].register_forward_hook(
                        self._modify_self_attn_qkv(layer_name)
                    )
                    self.attention_hooks.append(modify_hook)
                
                print(f"Hook registered for attention layer {i}: {layer_name}")
            else:
                print(f"Warning: Attention layer {i} not found. Available layers: 0-{len(attn)-1}")
        
        print(f"Target attention layers for injection: {self.attn_layers}")
        print(f"Total attention layers found: {len(attn)}")
    
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
    
    # TODO interpolate between content and style attention outputs using a guidance scale
    def _modify_self_attn_qkv(self, name):
        """Hook function to inject modified Q, K, V into attention modules.
        Adapted to match the diffusers implementation exactly."""
        def hook(model, input, output):
            if self.trigger_modify_qkv:
                # TODO 
                # 1. get current modified outputs (hidden states) 
                # 2. get stored features for modification (also hidden states, may have to modify the
                # self.attn_features_modify dict store to store the returned hidden states)
                # Get current features
                _, q_cs, k_cs, v_cs, _ = self._attention_op(model, input[0])
                
                # Get stored features for modification
                if name in self.attn_features_modify and int(self.cur_t) in self.attn_features_modify[name]:
                    q_c, k_s, v_s = self.attn_features_modify[name][int(self.cur_t)]
                    
                    # Handle batch size mismatch due to classifier-free guidance
                    # The current batch might be larger due to CFG (uncond + cond)
                    current_batch_size = q_cs.shape[0]
                    stored_batch_size = q_c.shape[0]
                    
                    if current_batch_size != stored_batch_size:
                        # Repeat stored features to match current batch size
                        repeat_factor = current_batch_size // stored_batch_size
                        q_c = q_c.repeat(repeat_factor, 1, 1)
                        k_s = k_s.repeat(repeat_factor, 1, 1)
                        v_s = v_s.repeat(repeat_factor, 1, 1)
                    
                    # query preservation
                    q_hat_cs = q_c * self.gamma + q_cs * (1 - self.gamma)

                    # NOTE k,v copied from style
                    # TODO  "A Training-Free Approach for Music Style Transfer with Latent Diffusion Models"
                    # uses a scaling factor between style and content attention outputs
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
    
    def _get_unet_layers(self, unet):
        """
        Get UNet layers for attention feature extraction.
        Adapted to match the diffusers implementation exactly.
        Returns (resnet_blocks, attention_blocks) where attention_blocks is a list of attention layers.
        """
        layer_num = [i for i in range(12)]  # Like in diffusers implementation
        resnet_layers = []
        attn_layers = []
        
        for idx, ln in enumerate(layer_num):
            up_block_idx = idx // 3
            layer_idx = idx % 3
            
            resnet_layers.append(getattr(unet, 'up_blocks')[up_block_idx].resnets[layer_idx])
            if up_block_idx > 0:
                attn_layers.append(getattr(unet, 'up_blocks')[up_block_idx].attentions[layer_idx])
            else:
                attn_layers.append(None)
        
        print(f"Found {len(attn_layers)} attention layers using diffusers method")
        return resnet_layers, attn_layers
    
    def _attention_op_embeddings(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, query=None, key=None, value=None, attention_probs=None, temperature=1.0):
        """
        Attention operation to get query, key, value and attention map from the UNet.
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
    
    def _attention_op_output_outputs_interp(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, query=None, key=None, value=None, attention_probs=None, temperature=1.0, guidance_scores=None):
        """
        Attention operation to get score outputs from the UNet.
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

        # NOTE scaling query
        query = query * temperature # same as applying it on qk matrix

        if attention_probs is None:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)

        batch_heads, img_len, txt_len = attention_probs.shape
        
        # 
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

    def _attention_op_residual_outputs(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, query=None, key=None, value=None, attention_probs=None, temperature=1.0):
        """
        Attention operation to get query, key, value and attention map from the UNet.
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
        query = query * temperature # same as applying it on qk matrix

        # q*k^T (where softmax is applied)
        if attention_probs is None:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)

        batch_heads, img_len, txt_len = attention_probs.shape
        
        # k*v^T 
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
    def extract_features_ddim(self, image, num_steps=50, save_feature_steps=50):
        """
        Extract features using DDIM inversion with memory optimization.
        Adapted to match the diffusers implementation.
        
        Args:
            image: Input image (content or style)
            num_steps: Number of DDIM inversion steps
            save_feature_steps: Number of steps to save features for
            
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

        # Encode image to latent
        if isinstance(image, Image.Image):
            image_tensor = preprocess_image(image).to(device=self.device, dtype=self.vae.dtype)
        else:
            image_tensor = image
            
        init_latent_dist = self.vae.encode(image_tensor).latent_dist
        init_latents = init_latent_dist.sample()
        init_latents = 0.18215 * init_latents

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
                
                # Predict noise - we need to provide encoder hidden states
                # For DDIM inversion, we can use a dummy embedding or skip the cross-attention
                batch_size = cur_latent.shape[0]
                hidden_size = self.text_encoder.config.hidden_size
                dummy_embedding = torch.zeros(batch_size, 77, hidden_size, device=cur_latent.device, dtype=cur_latent.dtype)
                
                noise_pred = self.unet(cur_latent, t.to(cur_latent.device), encoder_hidden_states=dummy_embedding).sample
                
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

    def styleid_riffuse(
        self,
        inputs: InferenceInput,
        content_image: Image.Image,
        style_image: Image.Image,
        use_adain_init: bool = True,
        use_attn_injection: bool = True,
        gamma: float = 0.75,
        T: float = 1.5,
        start_step: int = 49,
        attention_op_type: str = "1"
    ) -> Image.Image:
        """
        StyleID-enhanced riffusion inference for spectrogram style transfer.
        Adapted to match the diffusers implementation more closely.
        
        Args:
            inputs: Parameter dataclass
            content_image: Content spectrogram image
            style_image: Style spectrogram image  
            use_adain_init: Whether to use AdaIN initialization
            use_attn_injection: Whether to use attention feature injection
            gamma: Query preservation parameter
            T: Temperature scaling parameter
            start_step: Starting step for feature injection
        """
        self.gamma = gamma
        self.T = T
        self.start_step = start_step
        
        alpha = inputs.alpha
        start = inputs.start
        end = inputs.end

        guidance_scale = start.guidance * (1.0 - alpha) + end.guidance * alpha

        # Setup generators
        if self.device.lower().startswith("mps"):
            generator_start = torch.Generator(device="cpu").manual_seed(start.seed)
            generator_end = torch.Generator(device="cpu").manual_seed(end.seed)
        else:
            generator_start = torch.Generator(device=self.device).manual_seed(start.seed)
            generator_end = torch.Generator(device=self.device).manual_seed(end.seed)

        # NOTE wtf is this?
        ############### Text encodings with interpolation ###############
        embed_start = self.embed_text_weighted(start.prompt)
        embed_end = self.embed_text_weighted(end.prompt)
        text_embedding = embed_start + alpha * (embed_end - embed_start)

        # Setup feature extraction hooks
        self.setup_feature_extraction()
        
        # Extract content and style features - matching diffusers implementation
        print("Extracting content features...")
        self.trigger_get_qkv = True
        self.trigger_modify_qkv = False
        
        # Clear previous features
        self.attn_features = {}
        for i in self.attn_layers:
            layer_name = f"layer{i}_attn"
            self.attn_features[layer_name] = {}
        
        content_latents, _ = self.extract_features_ddim(
            content_image, 
            num_steps=inputs.num_inference_steps,
            save_feature_steps=inputs.num_inference_steps
        )
        content_features = copy.deepcopy(self.attn_features)
        
        print("Extracting style features...")
        # Clear features for style extraction
        self.attn_features = {}
        for i in self.attn_layers:
            layer_name = f"layer{i}_attn"
            self.attn_features[layer_name] = {}
        
        style_latents, _ = self.extract_features_ddim(
            style_image,
            num_steps=inputs.num_inference_steps, 
            save_feature_steps=inputs.num_inference_steps
        )
        style_features = copy.deepcopy(self.attn_features)
        
        # Cleanup attention hooks after feature extraction
        self.cleanup_attention_hooks()

        ############### AdaIN initialization ###############
        if use_adain_init:
            init_latents = adain(content_latents, style_latents)
        else:
            init_latents = content_latents

        ############### Merge features for injection - matching diffusers implementation ###############
        if use_attn_injection:
            # Set modify features like in diffusers implementation
            for layer_name in style_features.keys():
                self.attn_features_modify[layer_name] = {}
                for t in self.scheduler.timesteps:
                    t_val = t.item() if hasattr(t, 'item') else t
                    if t_val in content_features[layer_name] and t_val in style_features[layer_name]:
                        # content as q / style as kv - matching diffusers implementation
                        self.attn_features_modify[layer_name][t_val] = (
                            content_features[layer_name][t_val][0],  # content q
                            style_features[layer_name][t_val][1],    # style k
                            style_features[layer_name][t_val][2]     # style v
                        )
        else:
            self.attn_features_modify = {}

        # Setup for feature injection during generation
        self.trigger_get_qkv = False
        self.trigger_modify_qkv = use_attn_injection
        
        # Re-register hooks for injection phase
        if use_attn_injection:
            self._register_attention_hooks()
        
        # Run StyleID-enhanced interpolation
        outputs = self.styleid_interpolate_img2img(
            text_embeddings=text_embedding,
            init_latents=init_latents,
            generator_a=generator_start,
            generator_b=generator_end,
            interpolate_alpha=alpha,
            strength_a=start.denoising,
            strength_b=end.denoising,
            num_inference_steps=inputs.num_inference_steps,
            guidance_scale=guidance_scale,
            start_step=start_step,
        )

        return outputs["images"][0]

    @torch.no_grad()
    def styleid_interpolate_img2img(
        self,
        text_embeddings: torch.Tensor,
        init_latents: torch.Tensor,
        generator_a: torch.Generator,
        generator_b: torch.Generator,
        interpolate_alpha: float,
        mask: T.Optional[torch.Tensor] = None,
        strength_a: float = 0.8,
        strength_b: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: T.Optional[T.Union[str, T.List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: T.Optional[float] = 0.0,
        output_type: T.Optional[str] = "pil",
        start_step: int = 49,
        **kwargs,
    ):
        """
        StyleID-enhanced img2img interpolation with attention feature injection.
        Adapted to match the diffusers implementation.
        """
        batch_size = text_embeddings.shape[0]

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # Duplicate text embeddings for each generation per prompt
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # Classifier free guidance setup
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            if negative_prompt is None:
                uncond_tokens = [""]
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError("The length of `negative_prompt` should be equal to batch_size.")
            else:
                uncond_tokens = negative_prompt

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
            uncond_embeddings = uncond_embeddings.repeat_interleave(
                batch_size * num_images_per_prompt, dim=0
            )
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents_dtype = text_embeddings.dtype
        strength = (1 - interpolate_alpha) * strength_a + interpolate_alpha * strength_b

        # Get initial timestep
        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        timesteps = self.scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor(
            [timesteps] * batch_size * num_images_per_prompt, device=self.device
        )

        # NOTE initialise style and content latents
        # Add noise to latents
        noise_a = torch.randn(
            init_latents.shape, generator=generator_a, device=self.device, dtype=latents_dtype
        )
        noise_b = torch.randn(
            init_latents.shape, generator=generator_b, device=self.device, dtype=latents_dtype
        )
        noise = torch_util.slerp(interpolate_alpha, noise_a, noise_b)
        init_latents_orig = init_latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timesteps)

        # Prepare extra kwargs for scheduler
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        latents = init_latents.clone()
        t_start = max(num_inference_steps - init_timestep + offset, 0)
        timesteps = self.scheduler.timesteps[t_start:].to(self.device)

        # NOTE start styleid interpolation - matching diffusers implementation:
        for i, t in enumerate(self.progress_bar(timesteps)):
            # Set current timestep for attention hooks
            self.cur_t = t.item() if hasattr(t, 'item') else t
            
            # Skip injection before start_step
            if i < start_step:
                self.trigger_modify_qkv = False
            else:
                self.trigger_modify_qkv = True

            # Expand latents for classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Predict noise with StyleID injection
            noise_pred = self.unet(
                latent_model_input, 
                t, 
                encoder_hidden_states=text_embeddings
            ).sample

            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # Compute previous noisy sample
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # Apply mask if provided
            if mask is not None:
                init_latents_proper = self.scheduler.add_noise(
                    init_latents_orig, noise, torch.tensor([t])
                )
                latents = (init_latents_proper * mask) + (latents * (1 - mask))

        # Decode latents to image
        latents = 1.0 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return dict(images=image, latents=latents, nsfw_content_detected=False)

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: str,
        use_traced_unet: bool = True,
        channels_last: bool = False,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        local_files_only: bool = False,
        low_cpu_mem_usage: bool = False,
        cache_dir: T.Optional[str] = None,
    ) -> StyleIDRiffusionPipeline:
        """
        Load a StyleID Riffusion pipeline from a checkpoint.
        
        Args:
            checkpoint: Path to the model checkpoint
            use_traced_unet: Whether to use the traced unet for speedups
            device: Device to load the model on
            channels_last: Whether to use channels_last memory format
            local_files_only: Don't download, only use local files
            low_cpu_mem_usage: Attempt to use less memory on CPU
        """
        device = torch_util.check_device(device)

        if device == "cpu" or device.lower().startswith("mps"):
            print(f"WARNING: Falling back to float32 on {device}, float16 is unsupported")
            dtype = torch.float32

        # Try to load using the base pipeline's from_pretrained method
        # but handle the feature_extractor parameter manually


        try:
            # Load components individually to avoid the feature_extractor issue
            from diffusers import AutoencoderKL, UNet2DConditionModel
            from transformers import CLIPTextModel, CLIPTokenizer, CLIPFeatureExtractor
            from diffusers.schedulers import DDIMScheduler
            from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
            
            # Load components
            vae = AutoencoderKL.from_pretrained(
                checkpoint, 
                subfolder="vae",
                torch_dtype=dtype,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
            )
            
            text_encoder = CLIPTextModel.from_pretrained(
                checkpoint,
                subfolder="text_encoder",
                torch_dtype=dtype,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
            )
            
            tokenizer = CLIPTokenizer.from_pretrained(
                checkpoint,
                subfolder="tokenizer",
                local_files_only=local_files_only,
                cache_dir=cache_dir,
            )
            
            unet = UNet2DConditionModel.from_pretrained(
                checkpoint,
                subfolder="unet",
                torch_dtype=dtype,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
            )
            
            scheduler = DDIMScheduler.from_pretrained(
                checkpoint,
                subfolder="scheduler",
                local_files_only=local_files_only,
                cache_dir=cache_dir,
            )
            
            safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker",
                torch_dtype=dtype,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
            )
            
            feature_extractor = CLIPFeatureExtractor.from_pretrained(
                "CompVis/stable-diffusion-safety-checker",
                local_files_only=local_files_only,
                cache_dir=cache_dir,
            )
            
            # Create the StyleIDRiffusionPipeline
            styleid_pipeline = cls(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                # feature_extractor=feature_extractor,
            )
            

            # Move to device
            styleid_pipeline = styleid_pipeline.to(device)
            
            if channels_last:
                styleid_pipeline.unet.to(memory_format=torch.channels_last)

            return styleid_pipeline
            
        except Exception as e:
            print(f"Error loading components individually: {e}")
            # Fallback to the original method
            base_pipeline = RiffusionPipeline.load_checkpoint(
                checkpoint=checkpoint,
                use_traced_unet=use_traced_unet,
                channels_last=channels_last,
                dtype=dtype,
                device=device,
                local_files_only=local_files_only,
                low_cpu_mem_usage=low_cpu_mem_usage,
                cache_dir=cache_dir,
            )

            # Create StyleIDRiffusionPipeline with the same components
            styleid_pipeline = cls(
                vae=base_pipeline.vae,
                text_encoder=base_pipeline.text_encoder,
                tokenizer=base_pipeline.tokenizer,
                unet=base_pipeline.unet,
                scheduler=base_pipeline.scheduler,
                safety_checker=base_pipeline.safety_checker,
                # feature_extractor=base_pipeline.feature_extractor,
            )
            
            # Move to device
            styleid_pipeline = styleid_pipeline.to(device)

            return styleid_pipeline


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess an image for the model.
    """
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)

    image_np = np.array(image).astype(np.float32) / 255.0

    image_np = image_np[None].transpose(0, 3, 1, 2)

    image_torch = torch.from_numpy(image_np)
    return 2.0 * image_torch - 1.0


def preprocess_mask(mask: Image.Image, scale_factor: int = 8) -> torch.Tensor:
    """
    Preprocess a mask for the model.
    """
    # Convert to grayscale
    mask = mask.convert("L")

    # Resize to integer multiple of 32
    w, h = mask.size
    w, h = map(lambda x: x - x % 32, (w, h))
    mask = mask.resize((w // scale_factor, h // scale_factor), resample=Image.NEAREST)

    # Convert to numpy array and rescale
    mask_np = np.array(mask).astype(np.float32) / 255.0

    # Tile and transpose
    mask_np = np.tile(mask_np, (4, 1, 1))
    mask_np = mask_np[None].transpose(0, 1, 2, 3)

    # Invert to repaint white and keep black
    mask_np = 1 - mask_np

    return torch.from_numpy(mask_np) 