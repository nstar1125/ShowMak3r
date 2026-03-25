from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    PNDMScheduler,
    DDIMScheduler,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
)
# from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel -> deprecated
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import randn_tensor, is_compiled_module

from typing import Dict, Optional
from pathlib import Path
from copy import deepcopy

# suppress partial model loading warning
logging.set_verbosity_error()

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL.Image


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

def get_controlnet(name : str="inpaint_sd15", torch_dtype=torch.float32):
    if name == "inpaint_sd15":
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch_dtype)
    elif name == "openpose_sd15":
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch_dtype)
    elif name == "openpose_sd21":
        # https://huggingface.co/thibaud/controlnet-sd21-openpose-diffusers
        controlnet = ControlNetModel.from_pretrained("thibaud/controlnet-sd21-openpose-diffusers", torch_dtype=torch_dtype)
    else:
        raise NotImplementedError()
    
    return controlnet


class OPControlNet(nn.Module):
    def __init__(
        self,
        device,
        fp16=True,
        vram_O=False,
        sd_version="2.1", # "2.1", # '115'
        hf_key=None,
        # is_inpaint=False,
        t_range=[0.02, 0.98],
        textual_inversion_path: Optional[Dict]=None,
        do_cfg_rescale: bool=False,
        use_ti_free_prompt_on_controlnet: bool=False,
        textual_inversion_in_controlnet: bool=False
    ):
        super().__init__()

        self.device = device
        self.sd_version = sd_version
        # self.is_inpaint = is_inpaint
        self.do_cfg_rescale = do_cfg_rescale
        self.ti_in_controlnet = textual_inversion_in_controlnet
        self.use_ti_free_prompt_on_controlnet = use_ti_free_prompt_on_controlnet

        if hf_key is not None:
            print(f"[INFO] using hugging face custom model key: {hf_key}")
            model_key = hf_key
        elif self.sd_version == "2.1":
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == "2.0":
            raise NotImplementedError()
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == "1.5":
            model_key = "stable-diffusion-v1-5/stable-diffusion-v1-5" # runwayml/stable-diffusion-v1-5 deprecated
        else:
            raise ValueError(
                f"[ERROR] Stable-diffusion version {self.sd_version} not supported."
            )
        self.dtype = torch.float16 if fp16 else torch.float32


        if sd_version == "2.1":
            controlnets = [
                get_controlnet("openpose_sd21", self.dtype)
            ]
        else:
            controlnets = [
                get_controlnet("openpose_sd15", self.dtype)
            ]
        self.controlnet_conditioning_scale = [1.0]
        # TODO - deprecated
        # if self.is_inpaint:
        #     pass

        # Create model
        # if self.is_inpaint:
        #     raise NotImplementedError()
        # else:
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_key, 
            # vae=vae,
            controlnet=controlnets, 
            torch_dtype=self.dtype
        )

        self.inv_module_dict = dict()
        if textual_inversion_path is not None:
            self.inv_module_dict = dict()
            for pnum, inv_path in textual_inversion_path.items():
                print(f"[INFO] loading inversion_datasets from {str(inv_path)}")
                # TODO - deprecated
                # if self.is_inpaint:
                #     raise NotImplementedError()
                # else:
                _pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    model_key, 
                    controlnet=controlnets, 
                    torch_dtype=self.dtype
                )
                # load textual inversion weights
                atten_processor = load_attn_process(Path(str(inv_path)), weight_name="pytorch_custom_diffusion_weights.bin")
                for k, v in atten_processor.items():
                    atten_processor[k] = v.to(device)
                _pipe.load_textual_inversion(Path(str(inv_path)), weight_name="<new1>.bin")

                if is_xformers_available():
                    print(f"[INFO] Xformers is availabe! use memory efficient one")
                    _pipe.enable_xformers_memory_efficient_attention()

                if vram_O:
                    _pipe.enable_sequential_cpu_offload()
                    _pipe.enable_vae_slicing()
                    _pipe.unet.to(memory_format=torch.channels_last)
                    _pipe.enable_attention_slicing(1)
                    # pipe.enable_model_cpu_offload()
                else:
                    _pipe.to(device)

                self.inv_module_dict[pnum] = dict(
                    atten_processor = atten_processor,
                    tokenizer = _pipe.tokenizer,
                    text_encoder = _pipe.text_encoder,
                )

                del _pipe
        
        # load TI-free encoder / decoder
        if self.use_ti_free_prompt_on_controlnet and not self.ti_in_controlnet:
            print(f"[INFO] loading TI-free encoder and decoder...")
            _pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model_key, 
                controlnet=controlnets, 
                torch_dtype=self.dtype
            )
            if is_xformers_available():
                print(f"[INFO] Xformers is availabe! use memory efficient one")
                _pipe.enable_xformers_memory_efficient_attention()

            if vram_O:
                _pipe.enable_sequential_cpu_offload()
                _pipe.enable_vae_slicing()
                _pipe.unet.to(memory_format=torch.channels_last)
                _pipe.enable_attention_slicing(1)
                # pipe.enable_model_cpu_offload()
            else:
                _pipe.to(device)
                    
            self.ti_free_tokenizer = _pipe.tokenizer
            self.ti_free_text_encoder = _pipe.text_encoder
        else:
            self.ti_free_tokenizer = None
            self.ti_free_text_encoder = None

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype
        )

        self.controlnet = pipe.controlnet
        self.device = pipe._execution_device

        self.control_image_processor = pipe.control_image_processor         # (It automatically change PIL to rgb)
        self.image_processor = pipe.image_processor
        # self.pipe = pipe
        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_noise_level(t_range)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self._embeddings = None
        self._ti_free_embeddings = None
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)


    def set_noise_level(self, t_range):
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])


    def set_textual_inversion(self, pnum):
        _temp = dict()
        for k, v in self.inv_module_dict[pnum]['atten_processor'].items():
            _temp[k] = v

        if self.ti_in_controlnet:
            self.controlnet.nets[0].set_attn_processor(_temp)
            self.tokenizer = self.inv_module_dict[pnum]['tokenizer']
            self.text_encoder = self.inv_module_dict[pnum]['text_encoder']
        else:
            self.unet.set_attn_processor(_temp)
            self.tokenizer = self.inv_module_dict[pnum]['tokenizer']
            self.text_encoder = self.inv_module_dict[pnum]['text_encoder']

    @property
    def embeddings(self):
        if isinstance(self._embeddings, dict) and self.pnum in self._embeddings:
            return self._embeddings[self.pnum]
        else:
            return self._embeddings

    @property
    def ti_free_embeddings(self):
        assert self.use_ti_free_prompt_on_controlnet or self.ti_in_controlnet, "Invalid access"
        if isinstance(self._ti_free_embeddings, dict) and self.pnum in self._ti_free_embeddings:
            return self._ti_free_embeddings[self.pnum]
        else:
            return self._ti_free_embeddings

    def set_text_embeds(self, prompts, negative_prompts):
        self.prompts = prompts
        self.negative_prompts = negative_prompts
    
    
    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts, pnum=-1):
        # TODO: Currently, we calculate ALL TI every iteration, which is waste of computation.
        # We need to get person number and only updates target embeddings.
        self.prompts = prompts
        self.negative_prompts = negative_prompts
        if len(self.inv_module_dict) > 0:
            if int(pnum) >= 0:
                self.set_textual_inversion(pnum)
                pos_embeds = self.encode_text(prompts)  # [1, 77, 768]
                neg_embeds = self.encode_text(negative_prompts)
                self._embeddings[pnum] = torch.cat([neg_embeds, pos_embeds], dim=0)  # [2, 77, 768]
            else:
                # reset embeddings
                self._embeddings = dict()
                for pnum in (self.inv_module_dict.keys()):
                    self.set_textual_inversion(pnum)
                    pos_embeds = self.encode_text(prompts)  # [1, 77, 768]
                    neg_embeds = self.encode_text(negative_prompts)
                    self._embeddings[pnum] = torch.cat([neg_embeds, pos_embeds], dim=0)  # [2, 77, 768]
        else:
            pos_embeds = self.encode_text(prompts)  # [1, 77, 768]
            neg_embeds = self.encode_text(negative_prompts)
            self._embeddings = torch.cat([neg_embeds, pos_embeds], dim=0)  # [2, 77, 768]

        if self.ti_in_controlnet:
            ti_free_prompts = []
            for p in prompts:
                ti_free_prompts.append(p.replace("<new1>" + " ", ""))
            self.ti_free_prompts = ti_free_prompts
            self.ti_free_negative_prompts = negative_prompts
            
            if len(self.inv_module_dict) > 0:
                if int(pnum) >= 0:
                    self.set_textual_inversion(pnum)
                    pos_embeds = self.encode_text(self.ti_free_prompts)  # [1, 77, 768]
                    neg_embeds = self.encode_text(self.ti_free_negative_prompts)
                    self._ti_free_embeddings[pnum] = torch.cat([neg_embeds, pos_embeds], dim=0)  # [2, 77, 768]
                else:
                    self._ti_free_embeddings = dict()
                    for pnum in (self.inv_module_dict.keys()):
                        self.set_textual_inversion(pnum)
                        pos_embeds = self.encode_text(self.ti_free_prompts)  # [1, 77, 768]
                        neg_embeds = self.encode_text(negative_prompts)
                        self._ti_free_embeddings[pnum] = torch.cat([neg_embeds, pos_embeds], dim=0)  # [2, 77, 768]
            else:
                pos_embeds = self.encode_text(self.ti_free_prompts)  # [1, 77, 768]
                neg_embeds = self.encode_text(negative_prompts)
                self._ti_free_embeddings = torch.cat([neg_embeds, pos_embeds], dim=0)  # [2, 77, 768]
                
        elif self.use_ti_free_prompt_on_controlnet:
            ti_free_prompts = []
            for p in prompts:
                ti_free_prompts.append(p.replace("<new1>" + " ", ""))
            self.ti_free_prompts = ti_free_prompts
            self.ti_free_negative_prompts = negative_prompts
            
            # All person number share same encodings
            self.tokenizer = self.ti_free_tokenizer
            self.text_encoder = self.ti_free_text_encoder
            pos_embeds = self.encode_text(self.ti_free_prompts)  # [1, 77, 768]
            neg_embeds = self.encode_text(self.ti_free_negative_prompts)
            self._ti_free_embeddings = torch.cat([neg_embeds, pos_embeds], dim=0)  # [2, 77, 768]
                
    def encode_text(self, prompt):
        # prompt: [str]
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    @torch.no_grad()
    def refine(self, pred_rgb, pnum, cond_image=[], controlnet_weight=1.,
               guidance_scale=100, steps=50, strength=0.8, 
        ):

        self.pnum = pnum
        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)
        if len(self.inv_module_dict) > 0:
            self.set_textual_inversion(self.pnum)

        batch_size = pred_rgb.shape[0]
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(pred_rgb_512.to(self.dtype))
        # latents = torch.randn((1, 4, 64, 64), device=self.device, dtype=self.dtype)

        # Handle ControlNet part
        _cond_image = []
        for ci in cond_image:
            _cond_image.append(ci.resize((512,512)))
        cond_image = _cond_image

        #   Do controlnet conditioning
        guess_mode = False
        control_image = self.get_control_image(cond_image, pred_rgb_512, guess_mode)


        self.scheduler.set_timesteps(steps, device=torch.device("cuda:0"))
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.cuda()    ### I'm not sure why we need it
        init_step = int(steps * strength)
        latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
            t = t.reshape(-1).repeat(batch_size).cuda()
            latent_model_input = torch.cat([latents] * 2)
            tt = torch.cat([t] * 2)
            
            control_model_input = latent_model_input        # NOT ASSUMING GUESS MODE

            if self.use_ti_free_prompt_on_controlnet:
                controlnet_prompt_embeds = self.ti_free_embeddings.repeat(batch_size, 1, 1) 
            else:
                controlnet_prompt_embeds = self.embeddings.repeat(batch_size, 1, 1) 

            cond_scale = [cs * controlnet_weight for cs in self.controlnet_conditioning_scale] # Repeated define (for cap of multi-step version)
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                control_model_input,
                tt,
                encoder_hidden_states=controlnet_prompt_embeds,
                controlnet_cond=control_image,
                conditioning_scale=cond_scale,
                guess_mode=guess_mode,
                return_dict=False,
            )

            if self.ti_in_controlnet:
                unet_prompt_embeds = self.ti_free_embeddings.repeat(batch_size, 1, 1) 
            else:
                unet_prompt_embeds = self.embeddings.repeat(batch_size, 1, 1)

            noise_pred = self.unet(
                latent_model_input, 
                tt, 
                encoder_hidden_states=unet_prompt_embeds,
                cross_attention_kwargs=None,
                # ControlNet part
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            latents = latents.half()
        
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]
        return imgs

    def train_step(
        self,
        pred_rgb,
        loss_rgb=None,
        step_ratio=None,
        guidance_scale=100,
        as_latent=False,
        hold_last_img=False,
        cond_image=[],
        additional_prompt=None,
        pnum=-1,
        mask=None,
        cfg_rescale_weight=0.8, # referred to value used in avatar-studio # used 0.5 until Nov 10 8:54 PM
        controlnet_weight: float=1.0,
        minimum_mask_thrs = 0.02,
        **kwargs
    ):  
        '''
        ControlNet Guidance
        '''
        self.pnum = pnum
        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)
            
        # Get textual inversion
        if not (additional_prompt is None):
            prompts = [self.prompts[0] + additional_prompt]
            negative_prompts = self.negative_prompts
        else:
            prompts = prompts
            negative_prompts = self.negative_prompts
        self.get_text_embeds(prompts, negative_prompts, pnum=pnum)
        
        # Set TI (again)
        if len(self.inv_module_dict) > 0:
            self.set_textual_inversion(self.pnum)    

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode="bilinear", align_corners=False) * 2 - 1
            pred_rgb_512 = pred_rgb
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

            _cond_image = []
            for ci in cond_image:
                _cond_image.append(ci.resize((512,512)))
            cond_image = _cond_image

            if loss_rgb is not None:
                loss_rgb_512 = F.interpolate(loss_rgb, (512, 512), mode="bilinear", align_corners=False)
                # encode image into latents with vae, requires grad!
                loss_latents = self.encode_imgs(loss_rgb_512)
            else:
                loss_latents = latents

        if step_ratio is not None:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

        if mask is not None and False:
            print("[WARNING] single-step SDS. Applying HARD MASK with thresholding here")
            mask = mask.squeeze().unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mask, (512 // self.vae_scale_factor, 512 // self.vae_scale_factor), mode="bilinear", align_corners=False)
            mask = mask.to(device=latents.device, dtype=self.dtype)  

            thrs = (t / self.num_train_timesteps)  - minimum_mask_thrs
            mask = (mask > thrs)
        elif False:
            print("[WARNING] it's inpainting mode, but mask isn't quired here")
        
        # w(t), sigma_t^2
        w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)
            
            #   Do controlnet conditioning
            guess_mode = False
            control_model_input = latent_model_input        # NOT ASSUMING GUESS MODE
            if self.use_ti_free_prompt_on_controlnet:
                controlnet_prompt_embeds = self.ti_free_embeddings.repeat(batch_size, 1, 1) 
            else:
                controlnet_prompt_embeds = self.embeddings.repeat(batch_size, 1, 1) 
            cond_scale = [cs * controlnet_weight for cs in self.controlnet_conditioning_scale] # Repeated define (for cap of multi-step version)

            control_image = self.get_control_image(cond_image, pred_rgb_512, guess_mode)
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                control_model_input,
                tt,
                encoder_hidden_states=controlnet_prompt_embeds,
                controlnet_cond=control_image,
                conditioning_scale=cond_scale,
                guess_mode=guess_mode,
                return_dict=False,
            )

            # Do noise predictions
            noise_pred = self.unet(
                latent_model_input, 
                tt, 
                encoder_hidden_states=self.embeddings.repeat(batch_size, 1, 1),
                cross_attention_kwargs=None,
                # ControlNet part
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_pos - noise_pred_uncond
            )

            if self.do_cfg_rescale:
                std_pos = noise_pred_pos.std([1,2,3], keepdim=True)
                std_cfg = noise_pred.std([1,2,3], keepdim=True)

                rescale_factor = std_pos / std_cfg
                rescale_factor = cfg_rescale_weight*rescale_factor + (1-cfg_rescale_weight)
                noise_pred = noise_pred * rescale_factor

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        
        if not (mask is None):
            # simply filter out loss from masked region
            grad = grad * mask

        # seems important to avoid NaN...
        # grad = grad.clamp(-1, 1)

        target = (loss_latents - grad).detach()
        loss = 0.5 * F.mse_loss(loss_latents.float(), target, reduction='sum') / loss_latents.shape[0]

        if hold_last_img:
            if not (mask is None):
                target = (loss_latents - grad) * mask.float() +  loss_latents *(1-mask.float())
                target = target.clone().detach().half()
            else:
                target = target.clone().detach().half()
            self.last_loss_imgs = self.decode_latents(target)

            if not (mask is None):
                mask = mask.squeeze().detach().cpu().squeeze().numpy()
                mask = mask / mask.max()
                mask = (mask * 255).astype(np.uint8)
                mask = np.repeat(mask[...,None], 3, axis=-1)
                self.last_inpaint_masks = mask


            self.last_noise = self.decode_latents(noise.clone().detach().half())
            self.last_noised_img = self.decode_latents(latents_noisy.clone().detach().half())

        return loss, t

    def get_control_image(self, control_image, pred_rgb, guess_mode, do_classifier_free_guidance=True):
        height, width = self._default_height_width(None, None, pred_rgb)
        batch_size = pred_rgb.shape[0]
        num_images_per_prompt = 1

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet
        if isinstance(controlnet, ControlNetModel):
            control_image = control_image[0]    # load first image from list
            control_image = self.prepare_control_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=self.device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
        elif isinstance(controlnet, MultiControlNetModel):
            control_images = []

            for control_image_ in control_image:
                control_image_ = self.prepare_control_image(
                    image=control_image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=self.device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                control_images.append(control_image_)

            control_image = control_images
        else:
            assert False

        return control_image

    def prepare_control_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    def _default_height_width(self, height, width, image):
        # NOTE: It is possible that a list of images have different
        # dimensions for each image, so just checking the first image
        # is not _exactly_ correct, but it is simple.
        while isinstance(image, list):
            image = image[0]

        if height is None:
            if isinstance(image, PIL.Image.Image):
                height = image.height
            elif isinstance(image, torch.Tensor):
                height = image.shape[2]

            height = (height // 8) * 8  # round down to nearest multiple of 8

        if width is None:
            if isinstance(image, PIL.Image.Image):
                width = image.width
            elif isinstance(image, torch.Tensor):
                width = image.shape[3]

            width = (width // 8) * 8  # round down to nearest multiple of 8

        return height, width

    @torch.no_grad()
    def produce_latents(
        self,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if latents is None:
            latents = torch.randn(
                (
                    self.embeddings.shape[0] // 2,
                    self.unet.in_channels,
                    height // 8,
                    width // 8,
                ),
                device=self.device,
            )

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=self.embeddings
            ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def prompt_to_img(
        self,
        prompts,
        negative_prompts="",
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        self.get_text_embeds(prompts, negative_prompts)
        
        # Text embeds -> img latents
        latents = self.produce_latents(
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs




def load_attn_process(pretrained_model_name_or_path_or_dict,  **kwargs):
    import safetensors
    from collections import defaultdict
    from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
    from diffusers.utils import is_accelerate_available, _get_model_file #, DIFFUSERS_CACHE, HF_HUB_OFFLINE
    from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT, load_model_dict_into_meta
    from diffusers.models.attention_processor import CustomDiffusionAttnProcessor

    TEXT_ENCODER_NAME = "text_encoder"
    UNET_NAME = "unet"


    ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
    DIFFUSERS_CACHE = HUGGINGFACE_HUB_CACHE
    HF_HUB_OFFLINE = os.getenv("HF_HUB_OFFLINE", "").upper() in ENV_VARS_TRUE_VALUES

    LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"
    LORA_WEIGHT_NAME_SAFE = "pytorch_lora_weights.safetensors"

    TEXT_INVERSION_NAME = "learned_embeds.bin"
    TEXT_INVERSION_NAME_SAFE = "learned_embeds.safetensors"

    CUSTOM_DIFFUSION_WEIGHT_NAME = "pytorch_custom_diffusion_weights.bin"
    CUSTOM_DIFFUSION_WEIGHT_NAME_SAFE = "pytorch_custom_diffusion_weights.safetensors"


    cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
    force_download = kwargs.pop("force_download", False)
    # resume_download = kwargs.pop("resume_download", False) -> deprecated
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
    use_auth_token = kwargs.pop("use_auth_token", None)
    revision = kwargs.pop("revision", None)
    subfolder = kwargs.pop("subfolder", None)
    weight_name = kwargs.pop("weight_name", None)
    use_safetensors = kwargs.pop("use_safetensors", None)
    low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
    # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
    # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
    network_alphas = kwargs.pop("network_alphas", None)

    _pipeline = kwargs.pop("_pipeline", None)

    is_network_alphas_none = network_alphas is None

    allow_pickle = False

    if use_safetensors is None:
        use_safetensors = True
        allow_pickle = True

    user_agent = {
        "file_type": "attn_procs_weights",
        "framework": "pytorch",
    }

    if low_cpu_mem_usage and not is_accelerate_available():
        low_cpu_mem_usage = False



    model_file = None
    if not isinstance(pretrained_model_name_or_path_or_dict, dict):
        # Let's first try to load .safetensors weights
        if (use_safetensors and weight_name is None) or (
            weight_name is not None and weight_name.endswith(".safetensors")
        ):
            try:
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict,
                    weights_name=weight_name or LORA_WEIGHT_NAME_SAFE,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    # resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                )
                state_dict = safetensors.torch.load_file(model_file, device="cpu")
            except IOError as e:
                if not allow_pickle:
                    raise e
                # try loading non-safetensors weights
                pass
        if model_file is None:
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name or LORA_WEIGHT_NAME,
                cache_dir=cache_dir,
                force_download=force_download,
                # resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
            state_dict = torch.load(model_file, map_location="cpu")
    else:
        state_dict = pretrained_model_name_or_path_or_dict


    attn_processors = {}
    custom_diffusion_grouped_dict = defaultdict(dict)
    for key, value in state_dict.items():
        if len(value) == 0:
            custom_diffusion_grouped_dict[key] = {}
        else:
            if "to_out" in key:
                attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
            else:
                attn_processor_key, sub_key = ".".join(key.split(".")[:-2]), ".".join(key.split(".")[-2:])
            custom_diffusion_grouped_dict[attn_processor_key][sub_key] = value

    for key, value_dict in custom_diffusion_grouped_dict.items():
        if len(value_dict) == 0:
            attn_processors[key] = CustomDiffusionAttnProcessor(
                train_kv=False, train_q_out=False, hidden_size=None, cross_attention_dim=None
            )
        else:
            cross_attention_dim = value_dict["to_k_custom_diffusion.weight"].shape[1]
            hidden_size = value_dict["to_k_custom_diffusion.weight"].shape[0]
            train_q_out = True if "to_q_custom_diffusion.weight" in value_dict else False
            attn_processors[key] = CustomDiffusionAttnProcessor(
                train_kv=True,
                train_q_out=train_q_out,
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
            )
            attn_processors[key].load_state_dict(value_dict)


    return attn_processors



if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--negative", default="", type=str)
    parser.add_argument(
        "--sd_version",
        type=str,
        default="2.1",
        choices=["1.5", "2.0", "2.1"],
        help="stable diffusion version",
    )
    parser.add_argument(
        "--hf_key",
        type=str,
        default=None,
        help="hugging face Stable diffusion model key",
    )
    parser.add_argument("--fp16", action="store_true", help="use float16 for training")
    parser.add_argument(
        "--vram_O", action="store_true", help="optimization for low VRAM usage"
    )
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device("cuda")

    sd = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()
