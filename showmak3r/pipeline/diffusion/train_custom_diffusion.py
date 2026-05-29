#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Custom Diffusion authors and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import cv2
import argparse
import hashlib
import itertools
import json
import logging
import math
import os
import random
import shutil
import warnings
from pathlib import Path
from typing import Union, Callable

import numpy as np
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfApi, create_repo
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
    StableDiffusionControlNetPipeline,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import (
    CustomDiffusionAttnProcessor,
    CustomDiffusionAttnProcessor2_0,
    CustomDiffusionXFormersAttnProcessor,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available


from showmak3r.utils.draw_op_jnts import draw_op_img
from showmak3r.utils.jnts_utils import filter_invisible_joints
from showmak3r.utils.image_utils import img_add_text

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.22.0.dev0")
logger = get_logger(__name__)

def save_attn_procs(
        module,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
        **kwargs,
    ):
        r"""
        Save attention processor layers to a directory so that it can be reloaded with the
        [`~loaders.UNet2DConditionLoadersMixin.load_attn_procs`] method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save an attention processor to (will be created if it doesn't exist).
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful during distributed training and you
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main
                process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful during distributed training when you need to
                replace `torch.save` with another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or with `pickle`.

        Example:

        ```py
        import torch
        from diffusers import DiffusionPipeline

        pipeline = DiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16,
        ).to("cuda")
        pipeline.unet.load_attn_procs("path-to-save-model", weight_name="pytorch_custom_diffusion_weights.bin")
        pipeline.unet.save_attn_procs("path-to-save-model", weight_name="pytorch_custom_diffusion_weights.bin")
        ```
        """
        from diffusers.models.attention_processor import (
            CustomDiffusionAttnProcessor,
            CustomDiffusionAttnProcessor2_0,
            CustomDiffusionXFormersAttnProcessor,
        )
        TEXT_ENCODER_NAME = "text_encoder"
        UNET_NAME = "unet"

        LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"
        LORA_WEIGHT_NAME_SAFE = "pytorch_lora_weights.safetensors"

        CUSTOM_DIFFUSION_WEIGHT_NAME = "pytorch_custom_diffusion_weights.bin"
        CUSTOM_DIFFUSION_WEIGHT_NAME_SAFE = "pytorch_custom_diffusion_weights.safetensors"


        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if save_function is None:
            if safe_serialization:

                def save_function(weights, filename):
                    return safetensors.torch.save_file(weights, filename, metadata={"format": "pt"})

            else:
                save_function = torch.save

        os.makedirs(save_directory, exist_ok=True)

        is_custom_diffusion = any(
            isinstance(
                x,
                (CustomDiffusionAttnProcessor, CustomDiffusionAttnProcessor2_0, CustomDiffusionXFormersAttnProcessor),
            )
            for (_, x) in module.attn_processors.items()
        )
        if is_custom_diffusion:
            model_to_save = AttnProcsLayers(
                {
                    y: x
                    for (y, x) in module.attn_processors.items()
                    if isinstance(
                        x,
                        (
                            CustomDiffusionAttnProcessor,
                            CustomDiffusionAttnProcessor2_0,
                            CustomDiffusionXFormersAttnProcessor,
                        ),
                    )
                }
            )
            state_dict = model_to_save.state_dict()
            for name, attn in module.attn_processors.items():
                if len(attn.state_dict()) == 0:
                    state_dict[name] = {}
        else:
            model_to_save = AttnProcsLayers(module.attn_processors)
            state_dict = model_to_save.state_dict()

        if weight_name is None:
            if safe_serialization:
                weight_name = CUSTOM_DIFFUSION_WEIGHT_NAME_SAFE if is_custom_diffusion else LORA_WEIGHT_NAME_SAFE
            else:
                weight_name = CUSTOM_DIFFUSION_WEIGHT_NAME if is_custom_diffusion else LORA_WEIGHT_NAME

        # Save the model
        save_function(state_dict, os.path.join(save_directory, weight_name))
        logger.info(f"Model weights saved in {os.path.join(save_directory, weight_name)}")


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def save_model_card(repo_id: str, images=None, base_model=str, prompt=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
instance_prompt: {prompt}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- custom-diffusion
inference: true
---
    """
    model_card = f"""
# Custom Diffusion - {repo_id}

These are Custom Diffusion adaption weights for {base_model}. The weights were trained on {prompt} using [Custom Diffusion](https://www.cs.cmu.edu/~custom-diffusion). You can find some example images in the following. \n
{img_str}

\nFor more details on the training, please follow [this link](https://github.com/huggingface/diffusers/blob/main/examples/custom_diffusion).
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def collate_fn(examples, with_prior_preservation, prior_batch_size=1):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    cond_imgs = [example["cond_images"] for example in examples]
    mask = [example["mask"] for example in examples]
    data_mask = [example["data_mask"] for example in examples]
    prompts = [example["prompt"] for example in examples]
    
    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        if 'fb_instance_prompt_ids' in examples[0]:
            input_ids += [example["fb_instance_prompt_ids"] for example in examples[:prior_batch_size]]
            pixel_values += [example["fb_instance_images"] for example in examples[:prior_batch_size]]
            cond_imgs += [example["fb_cond_images"] for example in examples[:prior_batch_size]]
            mask += [example["fb_mask"] for example in examples[:prior_batch_size]]
            data_mask += [example["fb_data_mask"] for example in examples[:prior_batch_size]]
            prompts += [example["fb_prompt"] for example in examples[:prior_batch_size]]
        else:
            input_ids += [example["class_prompt_ids"] for example in examples[:prior_batch_size]]
            pixel_values += [example["class_images"] for example in examples[:prior_batch_size]]
            mask += [example["class_mask"] for example in examples[:prior_batch_size]]

    input_ids = torch.cat(input_ids, dim=0)
    pixel_values = torch.stack(pixel_values)
    cond_imgs = torch.stack(cond_imgs)
    mask = torch.stack(mask)
    data_mask = torch.cat(data_mask, dim=0)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    cond_imgs = cond_imgs.to(memory_format=torch.contiguous_format).float()
    mask = mask.to(memory_format=torch.contiguous_format).float()
    data_mask = data_mask.to(memory_format=torch.contiguous_format).float()

    batch = {
        "input_ids": input_ids, 
        "pixel_values": pixel_values, 
        "mask": mask.unsqueeze(1), 
        "data_mask": data_mask,
        "cond_imgs": cond_imgs,
        "raw_prompts": prompts,
        }
    
    
    # when raw-dataset exists
    if "raw_instance_images" in examples[0]:
        batch['raw_pixel_values'] = [example["raw_instance_images"].float().unsqueeze(0) for example in examples]
        batch['raw_mask'] = [example["raw_mask"].unsqueeze(0).unsqueeze(1).float() for example in examples]
        batch['raw_data_mask'] = [example["raw_data_mask"].float() for example in examples]
        
        # if with_prior_preservation and'fb_instance_prompt_ids' in examples[0]:
        #     raise NotImplementedError()
        
    elif "raw_mask" in examples[0]:
        raw_mask = [example["raw_mask"] for example in examples]
        raw_data_mask = [example["raw_data_mask"] for example in examples]
        raw_mask = torch.stack(raw_mask)
        raw_data_mask = torch.cat(raw_data_mask, dim=0)
        batch['raw_mask'] = raw_mask.to(memory_format=torch.contiguous_format).unsqueeze(1).float()
        batch['raw_data_mask'] = raw_data_mask.to(memory_format=torch.contiguous_format).float()
        
        # if with_prior_preservation and'fb_instance_prompt_ids' in examples[0]:
        #     raise NotImplementedError()
        
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class CustomDiffusionDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        concepts_list,
        tokenizer,
        size=512,
        mask_size=64,
        center_crop=False,
        with_prior_preservation=False,
        num_class_images=200,
        hflip=False,
        aug=True,
        color_jittering=False,
        random_bg=False,
        get_img_wo_resize=False,
        fullbody_prior_path=None,
        masking_cond_image=False,
        mask_cond_image_with_data_mask=False,
    ):
        self.get_img_wo_resize = get_img_wo_resize
        self.hflip = hflip
        self.size = size
        self.mask_size = mask_size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.interpolation = Image.BILINEAR
        self.aug = aug
        self.color_jittering = color_jittering
        self.random_bg = random_bg
        self.fullbody_prior_path = fullbody_prior_path
        self.use_fullbody_prior = False
        self.masking_cond_image = masking_cond_image
        self.mask_cond_image_with_data_mask = mask_cond_image_with_data_mask
        
        if fullbody_prior_path is not None:
            self.fullbody_img_dir = Path(fullbody_prior_path) / "op_imgs"
            self.fullbody_jnt_dir = Path(fullbody_prior_path) / "op_jnts"
            self.fullbody_prompt = f"photo of a person, entire body, high quality"
            self.use_fullbody_prior = True
                

        self.instance_images_path = []
        self.class_images_path = []
        self.with_prior_preservation = with_prior_preservation
        for concept in concepts_list:
            inst_img_path = [] 
            
            for x in Path(concept["instance_data_dir"]).iterdir():
                if x.is_file():
                    # Load conditions
                    if concept["instance_cond_dir"] is not None:
                        cond_fname = Path(concept["instance_cond_dir"]) / x.name
                        if not cond_fname.exists():
                            # We need to try "npy" joints case also.
                            cond_fname = Path(concept["instance_cond_dir"]) / f"{x.name.split('.')[0]}.npy"
                            if not cond_fname.exists():
                                print(f"[INFO] we cannot fine {str(cond_fname)} skipping")
                                continue
                    
                    
                    # Load view dependent prompts
                    if concept["view_prompt_dir"] is not None:
                        prompt_fname = Path(concept["view_prompt_dir"]) / x.name
                        if not prompt_fname.exists():
                            # We need to try "npy" joints case also.
                            prompt_fname = Path(concept["view_prompt_dir"]) / f"{x.name.split('.')[0]}.npy"
                            if not prompt_fname.exists():
                                prompt_fname = None
                    else:
                        prompt_fname = None
                                

                        
                    if x.name.split(".")[-1].lower() not in ['jpg', 'png', 'jpeg']:
                        print(f"skipping {x}")
                        continue
                    
                    inst_img_path.append((x, cond_fname, concept["instance_prompt"], prompt_fname))

            self.instance_images_path.extend(inst_img_path)

            if with_prior_preservation:
                if self.use_fullbody_prior:
                    self.fullbody_img_dict = dict()
                    for img_fname in self.fullbody_img_dir.glob("*.png"):
                        self.fullbody_img_dict[int(img_fname.name[:-4])] = img_fname
                        
                    self.fullbody_jnt_dict = dict()
                    for jnt_fname in self.fullbody_jnt_dir.glob("*.npy"):
                        self.fullbody_jnt_dict[int(jnt_fname.name[:-4])] = jnt_fname
                
                else:
                    class_data_root = Path(concept["class_data_dir"])
                    if os.path.isdir(class_data_root):
                        class_images_path = list(class_data_root.iterdir())
                        class_prompt = [concept["class_prompt"] for _ in range(len(class_images_path))]
                    else:
                        with open(class_data_root, "r") as f:
                            class_images_path = f.read().splitlines()
                        with open(concept["class_prompt"], "r") as f:
                            class_prompt = f.read().splitlines()
                    
                    if str(class_data_root).split("/")[0] != class_images_path[0].split("/")[1]:
                        _class_images_path = []
                        relative_dir = str(class_data_root).split(class_images_path[0].split("/")[1])[0]
                        for cip in class_images_path:
                            cip = cip[:2] + relative_dir + cip[2:]
                            _class_images_path.append(cip)
                        class_images_path = _class_images_path

                    class_img_path = list(zip(class_images_path, class_prompt))
                    self.class_images_path.extend(class_img_path[:num_class_images])

        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)
        self.random_flip = transforms.RandomHorizontalFlip(0.5 * hflip)
        self.deterministic_flip = transforms.RandomHorizontalFlip(1.)

        if self.color_jittering:
            print("[DEBUG] Use Color Jittering!")
            self.jitter = torchvision.transforms.ColorJitter(brightness=.1, hue=.1)

        self.image_transforms = transforms.Compose(
            [
                self.random_flip,
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def preprocess(self, image, cond_image, data_mask, scale, resample, bg_color: float=-1.):
        """
        We do spatial augmentation, by random cropping(zooming) or expanding image.
        
        - scale: value around (512//3 ~ 512*1.6). If scale > self.size, it means zooming in. 
        - If we use raw-image resolution, we should apply resize at final stage (Need totally different formulation)
        
        """
        if self.random_bg:
            np_mask = data_mask.clone().detach().squeeze().unsqueeze(-1).numpy()
            np_image = np.array(image)

            bg_color = (np.random.rand(3) * 255).astype(np.uint8)[None, None]
            np_image = np_image * np_mask + (1-np_mask) * bg_color

            np_image = np.array(np_image).astype(np.uint8)
            image = Image.fromarray(np_image)


        data_mask = F.interpolate(data_mask, (scale, scale), mode="bilinear", align_corners=False)
        
        outer, inner = self.size, scale
        factor = self.size // self.mask_size
        if scale > self.size:
            outer, inner = scale, self.size
        top, left = np.random.randint(0, outer - inner + 1), np.random.randint(0, outer - inner + 1)
        image = image.resize((scale, scale), resample=resample)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
                
            
        instance_image = np.ones((self.size, self.size, 3), dtype=np.float32) * bg_color
        latent_mask = np.zeros((self.size // factor, self.size // factor))
        if scale > self.size:
            instance_image = image[top : top + inner, left : left + inner, :]
            latent_mask = np.ones((self.size // factor, self.size // factor))
        else:
            instance_image[top : top + inner, left : left + inner, :] = image
            latent_mask[
                top // factor + 1 : (top + scale) // factor - 1, left // factor + 1 : (left + scale) // factor - 1
            ] = 1.0
        mask = cv2.resize(latent_mask, (self.size, self.size))
            
        # do for cond_image
        if cond_image is not None:
            cond_image = cond_image.resize((scale, scale), resample=resample)
            cond_image = np.array(cond_image).astype(np.uint8).astype(np.float32)
            cond_image = (cond_image / 255.)
            instance_cond = np.ones((self.size, self.size, 3), dtype=np.float32) * 0.
            if scale > self.size:
                instance_cond = cond_image[top : top + inner, left : left + inner, :]
            else:
                instance_cond[top : top + inner, left : left + inner, :] = cond_image
        else:
            instance_cond = None

        # do for data_mask
        instance_data_mask = torch.ones((1, 1, self.size, self.size), dtype=torch.float32).to(data_mask.device) * bg_color
        if scale > self.size:
            instance_data_mask = data_mask[..., top : top + inner, left : left + inner]
        else:
            instance_data_mask[..., top : top + inner, left : left + inner] = data_mask
        latent_instance_data_mask = F.interpolate(instance_data_mask, (self.mask_size, self.mask_size), mode="bilinear", align_corners=False)
        
        return instance_image, instance_cond, latent_instance_data_mask, latent_mask, instance_data_mask, mask
    
    
    def preprocess_wo_resize(self, image, cond_image, data_mask, scale, resample, bg_color: float=-1., is_cond_jnts=False):
        """
        We do spatial augmentation, by random cropping(zooming) or expanding image. (wo resizing)
        
        - scale: value around (512//3 ~ 512*1.6). If scale > self.size, it means zooming in. 
        - If we use raw-image resolution, we should apply resize at final stage (Need totally different formulation)
        """
        H, W = image.size
        factor = scale / self.size
        
        # Image data-type modification
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        
        if self.random_bg:
            np_mask = data_mask.clone().detach().squeeze().unsqueeze(-1).numpy()

            bg_color = (np.random.rand(3) * 2 - 1.).astype(np.float32)[None, None]
            image = image * np_mask + (1-np_mask) * bg_color
            

        if cond_image is not None:
            if is_cond_jnts:
                pass 
            else:
                cond_image = np.array(cond_image).astype(np.uint8).astype(np.float32)
                cond_image = (cond_image / 255.)
        else:
            instance_cond = None
            
        # Spatial Augmentations
        if factor > 1.:
            # zoom-in case
            outer = H
            inner = int(H / factor)
            top, left = np.random.randint(0, outer - inner + 1), np.random.randint(0, outer - inner + 1)
            
            instance_image = image[top : top + inner, left : left + inner, :]
            mask = np.ones((inner, inner))
            instance_data_mask = data_mask[..., top : top + inner, left : left + inner]
            
            if cond_image is not None and is_cond_jnts:
                # convert in new format
                new_jnts = []
                for jnt in cond_image:
                    if jnt is None:
                        new_jnts.append(jnt)
                    else:
                        new_x = (jnt[0] - left) * self.size / inner
                        new_y = (jnt[1] - top) * self.size / inner
                        new_jnt = [new_x, new_y]
                        new_jnts.append(new_jnt)
                
                # new_jnts = filter_invisible_joints(new_jnts)
                instance_cond = draw_op_img(new_jnts, self.size, output_type='cv2')
            else:
                instance_cond = cond_image[top : top + inner, left : left + inner, :]
                instance_cond = cv2.resize(instance_cond, (self.size, self.size))
            
        else:
            outer = int(H / factor)
            inner = H
            top, left = np.random.randint(0, outer - inner + 1), np.random.randint(0, outer - inner + 1)
            
            instance_image = np.ones((outer, outer, 3), dtype=np.float32) * bg_color            
            instance_image[top : top + inner, left : left + inner, :] = image
            
            mask = np.zeros((outer, outer))
            mask[top : top + inner, left : left + inner] = 1.0 
            
            instance_data_mask = torch.zeros((1, 1, outer, outer), dtype=torch.float32).to(data_mask.device)
            instance_data_mask[..., top : top + inner, left : left + inner] = data_mask
            
            if cond_image is not None and is_cond_jnts:
                # convert in new format
                new_jnts = []
                for jnt in cond_image:
                    if jnt is None:
                        new_jnts.append(jnt)
                    else:
                        new_x = (jnt[0] + left) * self.size / outer
                        new_y = (jnt[1] + top) * self.size / outer
                        new_jnt = [new_x, new_y]
                        new_jnts.append(new_jnt)
                
                # new_jnts = filter_invisible_joints(new_jnts)
                instance_cond = draw_op_img(new_jnts, self.size, output_type='cv2')
                cond_mask = cv2.resize(mask, (self.size, self.size))
                instance_cond[cond_mask==0] *= 0                                        # Masking invisible region's openpose condition
            else:
                instance_cond = np.ones((outer, outer, 3), dtype=np.float32) * 0.
                instance_cond[top : top + inner, left : left + inner, :] = cond_image
                instance_cond = cv2.resize(instance_cond, (self.size, self.size))
        
        # cv2.imwrite('test.jpg', instance_cond)
        
        instance_cond = np.array(instance_cond).astype(np.uint8).astype(np.float32)
        instance_cond = (instance_cond / 255.)
        instance_cond = torch.from_numpy(instance_cond).permute(2, 0, 1)
        
        latent_instance_data_mask = F.interpolate(instance_data_mask, (self.mask_size, self.mask_size), mode="bilinear", align_corners=False)
        latent_mask = cv2.resize(mask, (self.mask_size, self.mask_size))
        latent_mask = latent_mask.astype(np.float32)

        if self.masking_cond_image:
            _latent_mask = torch.from_numpy(latent_mask).squeeze()[None, None]
            instance_cond = instance_cond * F.interpolate(_latent_mask, (self.size, self.size), mode="bilinear", align_corners=False)[0]
            
        if self.mask_cond_image_with_data_mask:
            instance_cond = instance_cond * F.interpolate(instance_data_mask, (self.size, self.size), mode="bilinear", align_corners=False)[0]
        
        return instance_image, instance_cond, latent_instance_data_mask, latent_mask, instance_data_mask, mask

    def __getitem__(self, index):
        example = {}
        instance_image, cond_image, instance_prompt, prompt_fname = self.instance_images_path[index % self.num_instance_images]
        
        # We need to apply same transformation for all "image / cond / masks"
        # It's why we use random.random here instead of RandomHorizontalFlip here.
        do_flip = bool(random.uniform(0, self.hflip) > 0.5)
        
        instance_img = cv2.imread(str(instance_image), -1)
        if instance_img.shape[-1] == 4:
            # To support alpha-matted background, we divide by 255.
            data_mask = instance_img[..., -1].astype(np.float32) / 255
            data_mask[data_mask > 1] = 1
            data_mask = torch.from_numpy(data_mask).squeeze().unsqueeze(0).unsqueeze(0)
        else:
            data_mask = torch.ones(instance_img[..., -1].shape).squeeze().unsqueeze(0).unsqueeze(0)
        if do_flip:
            data_mask = self.deterministic_flip(data_mask)
        
        
        instance_image = Image.open(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
            
        if do_flip:
            instance_image = self.deterministic_flip(instance_image)
        
        if cond_image is not None:
            cond_is_npy = False
            if cond_image.name.split(".")[-1] == 'npy':
                cond_is_npy = True
                cond_jnts = np.load(cond_image, allow_pickle=True)
                if do_flip:
                    _, _, H, W = instance_image.shape
                    
                    # flip the joint locations
                    _cond_jnts = []
                    for jnt in cond_jnts:
                        if jnt is None or ((jnt[0] < 0) and jnt[1] < 0):
                            _cond_jnts.append(None)
                        else:
                            _cond_jnts.append(
                                [W-jnt[0], jnt[1]]
                            )
                    cond_jnts = _cond_jnts
                else:
                    _cond_jnts = []
                    for jnt in cond_jnts:
                        if jnt is None or ((jnt[0] < 0) and jnt[1] < 0):
                            _cond_jnts.append(None)
                        else:
                            _cond_jnts.append(
                                [jnt[0], jnt[1]]
                            )
                    cond_jnts = _cond_jnts
                    
                cond_image = cond_jnts
                    
            else:
                cond_image = Image.open(cond_image)
                if not cond_image.mode == "RGB":
                    cond_image = cond_image.convert("RGB")
                if do_flip:
                    cond_image = self.deterministic_flip(cond_image)    
        
        # apply resize augmentation and create a valid image region mask
        random_scale = self.size
        if self.aug:
            #### Modified values!
            random_scale = (
                np.random.randint(self.size // 3, self.size + 1)
                if np.random.uniform() < 0.66
                else np.random.randint(int(1.2 * self.size), int(1.6 * self.size))
            )
            
        if self.get_img_wo_resize:
            raw_instance_image, cond_image, latent_data_mask, latent_mask, data_mask, mask = self.preprocess_wo_resize(instance_image, cond_image, data_mask, random_scale, self.interpolation, is_cond_jnts=cond_is_npy)
            raw_instance_image = torch.from_numpy(raw_instance_image).permute(2, 0, 1)
            
            example["instance_images"] = F.interpolate(raw_instance_image[None], (self.size, self.size), mode="bilinear", align_corners=False)[0]
            example["cond_images"] = cond_image
            example["mask"] = torch.from_numpy(latent_mask)
            example['data_mask'] = latent_data_mask
            
            example["raw_instance_images"] = raw_instance_image
            example["raw_mask"] =  torch.from_numpy(mask)
            example["raw_data_mask"] = data_mask
        
        else:
            instance_image, cond_image, latent_data_mask, latent_mask, data_mask, mask = self.preprocess(instance_image, cond_image, data_mask, random_scale, self.interpolation)
            example["instance_images"] = torch.from_numpy(instance_image).permute(2, 0, 1)
            example["cond_images"] = torch.from_numpy(cond_image).permute(2, 0, 1)
            example["mask"] = torch.from_numpy(latent_mask)
            example['data_mask'] = latent_data_mask
            
            example["raw_mask"] = torch.from_numpy(mask)
            example['raw_data_mask'] = data_mask

        if random_scale < 0.6 * self.size:
            instance_prompt = np.random.choice(["a far away ", "very small "]) + instance_prompt
        elif random_scale < 0.8 * self.size:
            instance_prompt = np.random.choice(["small ", ""]) + instance_prompt
        elif random_scale > self.size:
            instance_prompt = np.random.choice(["zoomed in ", "close up "]) + instance_prompt
            
            
        # Load prompts (view-dependent)
        if not (prompt_fname is None):
            view_dependent_prompt = np.load(prompt_fname)[()]
            instance_prompt = instance_prompt + view_dependent_prompt
        
        
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        example['prompt'] = instance_prompt

        if self.with_prior_preservation:
            prior_aux_prompt = None
            if self.use_fullbody_prior:
                fb_idx = random.randint(0, len(self.fullbody_jnt_dict)-1)
                fid = sorted(list(self.fullbody_jnt_dict.keys()))[fb_idx]
                cond_image = self.fullbody_jnt_dict[fid]
                instance_image = self.fullbody_img_dict[fid]
                instance_prompt = self.fullbody_prompt
                
                # We need to apply same transformation for all "image / cond / masks"
                # It's why we use random.random here instead of RandomHorizontalFlip here.
                do_flip = bool(random.uniform(0, self.hflip) > 0.5)
                
                instance_img = cv2.imread(str(instance_image), -1)
                if instance_img.shape[-1] == 4:
                    data_mask = (instance_img[..., -1] > 0).astype(np.float32)
                    data_mask = torch.from_numpy(data_mask).squeeze().unsqueeze(0).unsqueeze(0)
                else:
                    data_mask = torch.ones(instance_img[..., -1].shape).squeeze().unsqueeze(0).unsqueeze(0)
                if do_flip:
                    data_mask = self.deterministic_flip(data_mask)
                
                
                instance_image = Image.open(instance_image)
                if not instance_image.mode == "RGB":
                    instance_image = instance_image.convert("RGB")
                    
                if do_flip:
                    instance_image = self.deterministic_flip(instance_image)
                
                if cond_image is not None:
                    cond_is_npy = False
                    if cond_image.name.split(".")[-1] == 'npy':
                        cond_is_npy = True
                        prior_data = np.load(cond_image, allow_pickle=True)[()]
                        cond_jnts = prior_data['jnts']
                        prior_aux_prompt = prior_data['prompts']
                        
                        if do_flip:
                            _, _, H, W = instance_image.shape
                            
                            # flip the joint locations
                            _cond_jnts = []
                            for jnt in cond_jnts:
                                if jnt is None or ((jnt[0] < 0) and jnt[1] < 0):
                                    _cond_jnts.append(None)
                                else:
                                    _cond_jnts.append(
                                        [W-jnt[0], jnt[1]]
                                    )
                            cond_jnts = _cond_jnts
                        else:
                            _cond_jnts = []
                            for jnt in cond_jnts:
                                if jnt is None or ((jnt[0] < 0) and jnt[1] < 0):
                                    _cond_jnts.append(None)
                                else:
                                    _cond_jnts.append(
                                        [jnt[0], jnt[1]]
                                    )
                            cond_jnts = _cond_jnts
                            
                        cond_image = cond_jnts
                    else:
                        cond_image = Image.open(cond_image)
                        if not cond_image.mode == "RGB":
                            cond_image = cond_image.convert("RGB")
                        if do_flip:
                            cond_image = self.deterministic_flip(cond_image)    
                
                # apply resize augmentation and create a valid image region mask
                random_scale = self.size
                if self.aug:
                    random_scale = (
                        np.random.randint(self.size // 1.5, self.size + 1)
                        if np.random.uniform() < 0.6
                        else np.random.randint(int(1.3 * self.size), int(2.0 * self.size))
                    )

                if self.get_img_wo_resize:
                    raw_instance_image, cond_image, latent_data_mask, latent_mask, data_mask, mask = self.preprocess_wo_resize(instance_image, cond_image, data_mask, random_scale, self.interpolation, is_cond_jnts=cond_is_npy)
                    raw_instance_image = torch.from_numpy(raw_instance_image).permute(2, 0, 1)
                    
                    example["fb_instance_images"] = F.interpolate(raw_instance_image[None], (self.size, self.size), mode="bilinear", align_corners=False)[0]
                    example["fb_cond_images"] = cond_image
                    example["fb_mask"] = torch.from_numpy(latent_mask)
                    example['fb_data_mask'] = latent_data_mask
                    
                    example["fb_raw_instance_images"] = raw_instance_image
                    example["fb_raw_mask"] =  torch.from_numpy(mask)
                    example["fb_raw_data_mask"] = data_mask
            
                    # instance_image, cond_image, data_mask, mask = self.preprocess_wo_resize(instance_image, cond_image, data_mask, random_scale, self.interpolation, is_cond_jnts=cond_is_npy)
                    # raw_instance_image = torch.from_numpy(instance_image).permute(2, 0, 1)
                    # raw_mask = torch.from_numpy(mask)
                    # raw_data_mask = data_mask
                    
                    # example["fb_instance_images"] = F.interpolate(raw_instance_image[None], (self.size, self.size), mode="bilinear", align_corners=False)[0]
                    # example["fb_cond_images"] = torch.from_numpy(cond_image).permute(2, 0, 1)
                    # example["fb_mask"] = F.interpolate(raw_mask[None,None], (self.size, self.size), mode="bilinear", align_corners=False).squeeze()
                    # example['fb_data_mask'] = F.interpolate(raw_data_mask, (self.size, self.size), mode="bilinear", align_corners=False)
                    
                    # example["fb_raw_instance_images"] = raw_instance_image
                    # example["fb_raw_mask"] = raw_mask
                    # example["fb_raw_data_mask"] = raw_data_mask
                
                else:
                    instance_image, cond_image, latent_data_mask, latent_mask, data_mask, mask = self.preprocess(instance_image, cond_image, data_mask, random_scale, self.interpolation)
                    example["fb_instance_images"] = torch.from_numpy(instance_image).permute(2, 0, 1)
                    example["fb_cond_images"] = torch.from_numpy(cond_image).permute(2, 0, 1)
                    example["fb_mask"] = torch.from_numpy(latent_mask)
                    example['fb_data_mask'] = latent_data_mask
                    
                    example["fb_raw_mask"] = torch.from_numpy(mask)
                    example['fb_raw_data_mask'] = data_mask

                if random_scale < 0.6 * self.size:
                    instance_prompt = np.random.choice(["a far away ", "very small "]) + instance_prompt
                elif random_scale < 0.8 * self.size:
                    instance_prompt = np.random.choice(["small ", ""]) + instance_prompt
                elif random_scale > self.size:
                    instance_prompt = np.random.choice(["zoomed in ", "close up "]) + instance_prompt

                if not (prior_aux_prompt is None):
                    instance_prompt = instance_prompt + prior_aux_prompt
                    
                example["fb_instance_prompt_ids"] = self.tokenizer(
                    instance_prompt,
                    truncation=True,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids
                example['fb_prompt'] = instance_prompt


                
            else:
                class_image, class_prompt = self.class_images_path[index % self.num_class_images]
                class_image = Image.open(class_image)
                if not class_image.mode == "RGB":
                    class_image = class_image.convert("RGB")
                example["class_images"] = self.image_transforms(class_image)
                example["class_mask"] = torch.ones_like(example["mask"])
                example["class_prompt_ids"] = self.tokenizer(
                    class_prompt,
                    truncation=True,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids

        if self.color_jittering:
            example["instance_images"] = self.jitter(example["instance_images"])

        return example


def save_new_embed(text_encoder, modifier_token_id, accelerator, args, output_dir, safe_serialization=True):
    """Saves the new token embeddings from the text encoder."""
    logger.info("Saving embeddings")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight
    for x, y in zip(modifier_token_id, args.modifier_token):
        learned_embeds_dict = {}
        learned_embeds_dict[y] = learned_embeds[x]
        filename = f"{output_dir}/{y}.bin"

        if safe_serialization:
            safetensors.torch.save_file(learned_embeds_dict, filename, metadata={"format": "pt"})
        else:
            torch.save(learned_embeds_dict, filename)





def load_default_train_opt(diffusion_model_name):
    parser = argparse.ArgumentParser(description="Custom Diffusion training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--instance_cond_dir",
        type=str,
        default=None,
        help="A folder containing the openpose condition data of instance images.",
    )
    parser.add_argument(
        "--view_prompt_dir",
        type=str,
        default=None,
        help="A folder containing the view dependent prompt pre-calculated.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=2,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--real_prior",
        default=False,
        action="store_true",
        help="real images as prior.",
    )
    parser.add_argument(
        "--fullbody_prior",
        default=False,
        action="store_true",
        help="real images + full body as prior.",
    )
    parser.add_argument("--fullbody_prior_path", type=str, default=None, help="The source of fullbody data.")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=200,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="custom-diffusion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument(
        "--prior_batch_size", type=int, default=1, help="Batch size (per device) for prior images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--freeze_model",
        type=str,
        default="crossattn_kv",
        choices=["crossattn_kv", "crossattn"],
        help="crossattn to enable fine-tuning of all params in the cross attention",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--concepts_list",
        type=str,
        default=None,
        help="Path to json containing multiple concepts, will overwrite parameters like instance_prompt, class_prompt, etc.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--modifier_token",
        type=str,
        default=None,
        help="A token to use as a modifier for the concept.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default="ktn+pll+ucd", help="A token to use as initializer word."
    )
    parser.add_argument("--hflip", action="store_true", help="Apply horizontal flip data augmentation.")
    parser.add_argument(
        "--noaug",
        action="store_true",
        help="Dont apply augmentation during data augmentation when this flag is enabled.",
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.",
    )
    
    
    ############### Additional Implementations
    parser.add_argument(
        "--use_controlnet",
        action="store_true",
        help="Whether apply openpose condition or not.",
    )
    parser.add_argument(
        "--use_color_jitter",
        action="store_true",
        help="Whether apply openpose condition or not.",
    )
    parser.add_argument(
        "--controlnet_mode",
        type=str,
        default="v0",
        help="Mode of ControlNet.",
    )
    parser.add_argument(
        "--controlnet_weight",
        type=float,
        default=1.0,
        help="Weight of ControlNet.",
    )
    parser.add_argument(
        "--save_intermediate_for_debug",
        action="store_true",
        help="Whether to save intermediate outputs for debugging.",
    )
    parser.add_argument(
        "--instance_prompt_wo_token",
        type=str,
        default="photo of a person",
        help="Prompt without token",
    )
    
    parser.add_argument(
        "--bg_loss_weight",
        type=float,
        default=1.,
        help="Weight of loss that calculated from BG region of latents",
    )
    parser.add_argument(
        "--cd_only_on_controlnet",
        action="store_true",
        help="If True, apply 'learnable' distinct learnable Q-V in CustomDiffusion.",
    )
    parser.add_argument(
        "--random_bg",
        action="store_true",
        help="If True, apply random BG during data preprocessing using mask",
    )
    
    #### Image space loss ####
    parser.add_argument(
        "--image_space_loss",
        action="store_true",
        help="If True, apply loss after applying VAE decoder to convert in image space",
    )
    parser.add_argument(
        "--loss_in_original_img_resolution",
        action="store_true",
        help="If True, apply loss in raw image resolution",
    )
    parser.add_argument(
        "--get_img_wo_resize",
        action="store_true",
        help="If True, use better conditioning approach",
    )
    parser.add_argument(
        "--masking_cond_image",
        action="store_true",
        help="If True, Apply masking on the cond image (in case of using spatial augmentation)",
    )
    parser.add_argument(
        "--mask_cond_image_with_data_mask",
        action="store_true",
        help="If True, Apply masking on the cond image (in case of using data mask)",
    )
    
    
    
    
    
    
    args = parser.parse_args(["--pretrained_model_name_or_path", diffusion_model_name])

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.concepts_list is None:
            if args.class_data_dir is None:
                raise ValueError("You must specify a data directory for class images.")
            if args.class_prompt is None:
                raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    return args


def train_cd(args, wandb_exp_name=None):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    if args.report_to == "wandb":
        if accelerator.is_main_process:
            if not is_wandb_available():
                raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
            import wandb
            
            # set project name
            wandb_exp_name = "default" if wandb_exp_name is None else wandb_exp_name
            wandb.init(
                project='ShowMak3r_diffusion',
                name=wandb_exp_name
            )
            wandb.config.update(args)
            
    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("custom-diffusion", config=vars(args))

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    if args.concepts_list is None:
        args.concepts_list = [
            {
                "instance_prompt": args.instance_prompt,
                "class_prompt": args.class_prompt,
                "instance_data_dir": args.instance_data_dir,
                "class_data_dir": args.class_data_dir,
                "instance_cond_dir": args.instance_cond_dir,
                "view_prompt_dir": args.view_prompt_dir
            }
        ]
    else:
        with open(args.concepts_list, "r") as f:
            args.concepts_list = json.load(f)

    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        if args.fullbody_prior:
            assert Path(args.fullbody_prior_path).exists(), f"{str(args.fullbody_prior_path)} not exists"
        else:
            for i, concept in enumerate(args.concepts_list):
                class_images_dir = Path(concept["class_data_dir"])
                if not class_images_dir.exists():
                    class_images_dir.mkdir(parents=True, exist_ok=True)
                if args.real_prior:
                    assert (
                        class_images_dir / "images"
                    ).exists(), f"Please run: python retrieve.py --class_prompt \"{concept['class_prompt']}\" --class_data_dir {class_images_dir} --num_class_images {args.num_class_images}"
                    assert (
                        len(list((class_images_dir / "images").iterdir())) == args.num_class_images
                    ), f"Please run: python retrieve.py --class_prompt \"{concept['class_prompt']}\" --class_data_dir {class_images_dir} --num_class_images {args.num_class_images}"
                    assert (
                        class_images_dir / "caption.txt"
                    ).exists(), f"Please run: python retrieve.py --class_prompt \"{concept['class_prompt']}\" --class_data_dir {class_images_dir} --num_class_images {args.num_class_images}"
                    assert (
                        class_images_dir / "images.txt"
                    ).exists(), f"Please run: python retrieve.py --class_prompt \"{concept['class_prompt']}\" --class_data_dir {class_images_dir} --num_class_images {args.num_class_images}"
                    concept["class_prompt"] = os.path.join(class_images_dir, "caption.txt")
                    concept["class_data_dir"] = os.path.join(class_images_dir, "images.txt")
                    args.concepts_list[i] = concept
                    accelerator.wait_for_everyone()  
                else:
                    cur_class_images = len(list(class_images_dir.iterdir()))

                    if cur_class_images < args.num_class_images:
                        torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
                        if args.prior_generation_precision == "fp32":
                            torch_dtype = torch.float32
                        elif args.prior_generation_precision == "fp16":
                            torch_dtype = torch.float16
                        elif args.prior_generation_precision == "bf16":
                            torch_dtype = torch.bfloat16
                        pipeline = DiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            torch_dtype=torch_dtype,
                            safety_checker=None,
                            revision=args.revision,
                        )
                        pipeline.set_progress_bar_config(disable=True)

                        num_new_images = args.num_class_images - cur_class_images
                        logger.info(f"Number of class images to sample: {num_new_images}.")

                        sample_dataset = PromptDataset(args.class_prompt, num_new_images)
                        sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

                        sample_dataloader = accelerator.prepare(sample_dataloader)
                        pipeline.to(accelerator.device)

                        for example in tqdm(
                            sample_dataloader,
                            desc="Generating class images",
                            disable=not accelerator.is_local_main_process,
                        ):
                            images = pipeline(example["prompt"]).images

                            for i, image in enumerate(images):
                                hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                                image_filename = (
                                    class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                                )
                                image.save(image_filename)

                        del pipeline
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, accelerator=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
            use_fast=False,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    # noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # Adding a modifier token which is optimized ####
    # Code taken from https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py
    modifier_token_id = []
    initializer_token_id = []
    if args.modifier_token is not None:
        args.modifier_token = args.modifier_token.split("+")
        args.initializer_token = args.initializer_token.split("+")
        if len(args.modifier_token) > len(args.initializer_token):
            raise ValueError("You must specify + separated initializer token for each modifier token.")
        for modifier_token, initializer_token in zip(
            args.modifier_token, args.initializer_token[: len(args.modifier_token)]
        ):
            # Add the placeholder token in tokenizer
            num_added_tokens = tokenizer.add_tokens(modifier_token)
            if num_added_tokens == 0:
                raise ValueError(
                    f"The tokenizer already contains the token {modifier_token}. Please pass a different"
                    " `modifier_token` that is not already in the tokenizer."
                )

            # Convert the initializer_token, placeholder_token to ids
            token_ids = tokenizer.encode([initializer_token], add_special_tokens=False)
            print(token_ids)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")

            initializer_token_id.append(token_ids[0])
            modifier_token_id.append(tokenizer.convert_tokens_to_ids(modifier_token))

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = text_encoder.get_input_embeddings().weight.data
        for x, y in zip(modifier_token_id, initializer_token_id):
            token_embeds[x] = token_embeds[y]

        # Freeze all parameters except for the token embeddings in text encoder
        params_to_freeze = itertools.chain(
            text_encoder.text_model.encoder.parameters(),
            text_encoder.text_model.final_layer_norm.parameters(),
            text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        freeze_params(params_to_freeze)
    ########################################################
    ########################################################

    vae.requires_grad_(False)
    if args.modifier_token is None:
        text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    if accelerator.mixed_precision != "fp16" and args.modifier_token is not None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    
    
    ########################################################
    # LOAD CONTROLNET DATA
    
    if args.use_controlnet:
        print(f"\n\n[DEBUG] Use CONTROLNET on customdiffusion training\n\n")
        
        from diffusers import ControlNetModel
        controlnet_op = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=weight_dtype)
        controlnet_op.to(accelerator.device)
    else:
        controlnet_op = None
    ########################################################
    
    

    attention_class = (
        CustomDiffusionAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else CustomDiffusionAttnProcessor
    )
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            attention_class = CustomDiffusionXFormersAttnProcessor
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # now we will add new Custom Diffusion weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers

    # Only train key, value projection layers if freeze_model = 'crossattn_kv' else train all params in the cross attention layer
    train_kv = True
    train_q_out = False if args.freeze_model == "crossattn_kv" else True

    if not args.cd_only_on_controlnet:
        custom_diffusion_attn_procs = {}

        st = unet.state_dict()
        for name, _ in unet.attn_processors.items():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_custom_diffusion.weight": st[layer_name + ".to_k.weight"],
                "to_v_custom_diffusion.weight": st[layer_name + ".to_v.weight"],
            }
            if train_q_out:
                weights["to_q_custom_diffusion.weight"] = st[layer_name + ".to_q.weight"]
                weights["to_out_custom_diffusion.0.weight"] = st[layer_name + ".to_out.0.weight"]
                weights["to_out_custom_diffusion.0.bias"] = st[layer_name + ".to_out.0.bias"]
            if cross_attention_dim is not None:
                custom_diffusion_attn_procs[name] = attention_class(
                    train_kv=train_kv,
                    train_q_out=train_q_out,
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                ).to(unet.device)
                custom_diffusion_attn_procs[name].load_state_dict(weights)
            else:
                custom_diffusion_attn_procs[name] = attention_class(
                    train_kv=False,
                    train_q_out=False,
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                )
        del st
        unet.set_attn_processor(custom_diffusion_attn_procs)
        custom_diffusion_layers = AttnProcsLayers(unet.attn_processors)
        accelerator.register_for_checkpointing(custom_diffusion_layers)

    else:
        op_custom_diffusion_attn_procs = {}
        st = controlnet_op.state_dict()
        for name, _ in controlnet_op.attn_processors.items():
            cross_attention_dim = None if name.endswith("attn1.processor") else controlnet_op.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = controlnet_op.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(controlnet_op.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = controlnet_op.config.block_out_channels[block_id]
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_custom_diffusion.weight": st[layer_name + ".to_k.weight"],
                "to_v_custom_diffusion.weight": st[layer_name + ".to_v.weight"],
            }
            if train_q_out:
                weights["to_q_custom_diffusion.weight"] = st[layer_name + ".to_q.weight"]
                weights["to_out_custom_diffusion.0.weight"] = st[layer_name + ".to_out.0.weight"]
                weights["to_out_custom_diffusion.0.bias"] = st[layer_name + ".to_out.0.bias"]
            if cross_attention_dim is not None:
                op_custom_diffusion_attn_procs[name] = attention_class(
                    train_kv=train_kv,
                    train_q_out=train_q_out,
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                ).to(unet.device)
                op_custom_diffusion_attn_procs[name].load_state_dict(weights)
            else:
                op_custom_diffusion_attn_procs[name] = attention_class(
                    train_kv=False,
                    train_q_out=False,
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                )
        del st


        controlnet_op.set_attn_processor(op_custom_diffusion_attn_procs)
        op_custom_diffusion_layers = AttnProcsLayers(controlnet_op.attn_processors)
        accelerator.register_for_checkpointing(op_custom_diffusion_layers)


    
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.modifier_token is not None:
            text_encoder.gradient_checkpointing_enable()
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        if args.with_prior_preservation:
            args.learning_rate = args.learning_rate * 2.0

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    if args.cd_only_on_controlnet:
        ######## <0109> ControlNet CustomDiffusion  ########
        if args.modifier_token is not None:
            opt_targets = itertools.chain(
                text_encoder.get_input_embeddings().parameters(), 
                op_custom_diffusion_layers.parameters()
                )
        else:
            opt_targets = op_custom_diffusion_layers.parameters()
    else:
        # Simple Approach
        opt_targets = itertools.chain(text_encoder.get_input_embeddings().parameters(), custom_diffusion_layers.parameters()) \
                        if args.modifier_token is not None \
                        else custom_diffusion_layers.parameters()

    optimizer = optimizer_class(
        opt_targets,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = CustomDiffusionDataset(
        concepts_list=args.concepts_list,
        tokenizer=tokenizer,
        with_prior_preservation=args.with_prior_preservation,
        size=args.resolution,
        mask_size=vae.encode(
            torch.randn(1, 3, args.resolution, args.resolution).to(dtype=weight_dtype).to(accelerator.device)
        )
        .latent_dist.sample()
        .size()[-1],
        center_crop=args.center_crop,
        num_class_images=args.num_class_images,
        hflip=args.hflip,
        aug=not args.noaug,
        color_jittering=args.use_color_jitter,
        random_bg=args.random_bg,
        get_img_wo_resize=(args.get_img_wo_resize or args.loss_in_original_img_resolution),
        fullbody_prior_path=args.fullbody_prior_path,
        masking_cond_image=args.masking_cond_image,
        mask_cond_image_with_data_mask=args.mask_cond_image_with_data_mask
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation, args.prior_batch_size),
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    if args.cd_only_on_controlnet:
        if args.modifier_token is not None:
            op_custom_diffusion_layers, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                op_custom_diffusion_layers, text_encoder, optimizer, train_dataloader, lr_scheduler
            )
        else:
            op_custom_diffusion_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                op_custom_diffusion_layers, optimizer, train_dataloader, lr_scheduler
            )
    else:
        if args.modifier_token is not None:
            custom_diffusion_layers, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                custom_diffusion_layers, text_encoder, optimizer, train_dataloader, lr_scheduler
            )
        else:
            custom_diffusion_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                custom_diffusion_layers, optimizer, train_dataloader, lr_scheduler
            )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0


    # Load token here
    if args.use_controlnet and args.controlnet_mode in ["v2", "v4"]:
        with torch.no_grad():
            wo_ti_prompt_ids = train_dataset.tokenizer(
                args.instance_prompt_wo_token,
                truncation=True,
                padding="max_length",
                max_length=train_dataset.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

            wo_ti_prompt_ids = torch.cat([wo_ti_prompt_ids for _ in range(args.train_batch_size)], dim=0)
            wo_ti_prompt_ids = wo_ti_prompt_ids.to(text_encoder.device)
            wo_ti_prompt_embeds = text_encoder(wo_ti_prompt_ids)[0]

            print("Using ControlNet V2")


    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.modifier_token is not None:
            text_encoder.train()

        if args.image_space_loss:
            vae.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet), accelerator.accumulate(text_encoder):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                
                
                if args.use_controlnet:
                    cond_imgs = batch['cond_imgs']
                    controlnet_dtype = controlnet_op.dtype
                    cond_imgs = cond_imgs.to(dtype=controlnet_dtype).cuda()
                    
                    controlnet_prompt_embeds = encoder_hidden_states

                    if args.controlnet_mode == "v0":
                        # TI token disconnected from ControlNet
                        with torch.no_grad():
                            down_block_res_samples, mid_block_res_sample = controlnet_op(
                                noisy_latents,
                                timesteps,
                                encoder_hidden_states=controlnet_prompt_embeds,
                                controlnet_cond=cond_imgs,
                                conditioning_scale=args.controlnet_weight,
                                guess_mode=False,
                                return_dict=False,
                            )
                        model_pred = unet(
                            noisy_latents, 
                            timesteps, 
                            encoder_hidden_states=encoder_hidden_states,
                            cross_attention_kwargs=None,
                            # ControlNet part
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                            guess_mode=False,
                            return_dict=False,
                        )
                        model_pred = unet(
                            noisy_latents, 
                            timesteps, 
                            encoder_hidden_states=encoder_hidden_states,
                            cross_attention_kwargs=None,
                            # ControlNet part
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                        ).sample

                    elif args.controlnet_mode == "v2":
                        b_size = len(latents)//2 if args.with_prior_preservation else len(latents)
                        if args.with_prior_preservation:
                            controlnet_prompt_embeds = controlnet_prompt_embeds.clone().detach().to(wo_ti_prompt_embeds.device)
                            controlnet_prompt_embeds[:b_size] = wo_ti_prompt_embeds[:b_size]
                        else:
                            controlnet_prompt_embeds = wo_ti_prompt_embeds[:b_size]

                        # No TI token on controlnet
                        with torch.no_grad():
                            down_block_res_samples, mid_block_res_sample = controlnet_op(
                                noisy_latents,
                                timesteps,
                                encoder_hidden_states=controlnet_prompt_embeds,
                                controlnet_cond=cond_imgs,
                                conditioning_scale=1.,
                                guess_mode=False,
                                return_dict=False,
                            )
                        model_pred = unet(
                            noisy_latents, 
                            timesteps, 
                            encoder_hidden_states=encoder_hidden_states,
                            cross_attention_kwargs=None,
                            # ControlNet part
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                        ).sample

                    elif args.controlnet_mode == "v3":
                        raise NotImplementedError()
                    
                    elif args.controlnet_mode == "v4":
                        b_size = len(latents)//2 if args.with_prior_preservation else len(latents)

                        if args.with_prior_preservation:
                            unet_prompt_embeds = encoder_hidden_states.clone().detach().to(wo_ti_prompt_embeds.device)
                            unet_prompt_embeds[:b_size] = wo_ti_prompt_embeds[:b_size]
                        else:
                            unet_prompt_embeds = wo_ti_prompt_embeds[:b_size]
                        


                        # TI token connected with ControlNet
                        down_block_res_samples, mid_block_res_sample = controlnet_op(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states=controlnet_prompt_embeds,
                            controlnet_cond=cond_imgs,
                            conditioning_scale=1.,
                            guess_mode=False,
                            return_dict=False,
                        )
                        model_pred = unet(
                            noisy_latents, 
                            timesteps, 
                            encoder_hidden_states=unet_prompt_embeds,
                            cross_attention_kwargs=None,
                            # ControlNet part
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                        ).sample

                    else:
                        raise NotImplementedError(f"Invalid controlnet mode: {args.controlnet_mode}")
                    
                else:
                    # Predict the noise residual
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                
                # Convert in image space (if needed)
                mask_key = "mask"
                data_mask_key = "data_mask"
                if args.image_space_loss:
                    # Convert to Z_0
                    # Here we use DDIM style denoising.
                    prev_timestep = 0
                    alpha_prod_t = noise_scheduler.alphas_cumprod[timesteps].reshape(-1, 1, 1, 1)
                    beta_prod_t = 1 - alpha_prod_t
                    alpha_prod_t = alpha_prod_t.to(model_pred.device)
                    beta_prod_t = beta_prod_t.to(model_pred.device)


                    if noise_scheduler.config.prediction_type == "epsilon":
                        pred_original_sample = (noisy_latents - beta_prod_t ** (0.5) * model_pred) / alpha_prod_t ** (0.5)
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        pred_original_sample = (alpha_prod_t**0.5) * noisy_latents - (beta_prod_t**0.5) * model_pred

                    model_pred = pred_original_sample
                    # model_pred = noise_scheduler.step(model_pred, timesteps, noisy_latents, **extra_step_kwargs, return_dict=False)[0]
                    
                    model_pred = 1 / vae.config.scaling_factor * model_pred
                    pred_imgs = vae.decode(model_pred).sample                 # (B, 4, 64, 64)
                    # pred_imgs = (pred_imgs / 2 + 0.5).clamp(0,1)            # (B, 3, 512, 512)
                    # pred_imgs = pred_imgs.clamp(-1,1)                       # (B, 3, 512, 512)
                    model_pred = pred_imgs
                    mask_key = "raw_mask"
                    data_mask_key = "raw_data_mask"
                     
                    
                if args.loss_in_original_img_resolution:
                    # Resized Case
                    assert 'raw_pixel_values' in batch, "raw pixel values have error on pre-processing"
                    target = [rpv.to(dtype=weight_dtype) for rpv in batch["raw_pixel_values"]]
                    
                    if args.with_prior_preservation and not args.fullbody_prior:
                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        _target = batch["pixel_values"].to(dtype=weight_dtype) # (B, 3, 512, 512)
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        _, target_prior = torch.chunk(_target, 2, dim=0)
                        
                        # Compute prior loss
                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                        
                        # Compute instance loss
                        loss = 0
                        for _model_pred, r_target, r_mask, r_data_mask in zip(model_pred, target, batch['raw_mask'], batch['raw_data_mask']):
                            mask = r_mask * (args.bg_loss_weight*(1 - r_data_mask) + r_data_mask)
                            raw_size = r_target.shape[-1]
                            _model_pred = F.interpolate(_model_pred[None], (raw_size, raw_size), mode="bilinear", align_corners=False)
                            
                            _loss = F.mse_loss(_model_pred.float(), r_target.float(), reduction="none")
                            _loss = ((_loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()
                            loss += _loss

                        # Add the prior loss to the instance loss.
                        loss = loss + args.prior_loss_weight * prior_loss
                    else:
                        # Compute instance loss
                        loss = 0
                        for _model_pred, r_target, r_mask, r_data_mask in zip(model_pred, target, batch['raw_mask'], batch['raw_data_mask']):
                            mask = r_mask * (args.bg_loss_weight*(1 - r_data_mask) + r_data_mask)
                            raw_size = r_target.shape[-1]
                            _model_pred = F.interpolate(_model_pred[None], (raw_size, raw_size), mode="bilinear", align_corners=False)
                            
                            _loss = F.mse_loss(_model_pred.float(), r_target.float(), reduction="none")
                            _loss = ((_loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()
                            loss += _loss
                        
                    
                else:
                    # Original CustomDiffusion Settings
                    # target = batch["pixel_values"].to(dtype=weight_dtype) # (B, 3, 512, 512)
                    
                    if args.with_prior_preservation and not args.fullbody_prior:
                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)
                        mask = torch.chunk(batch[mask_key], 2, dim=0)[0] * (args.bg_loss_weight*(1 - batch[data_mask_key]) + batch[data_mask_key])
                        # Compute instance loss
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        loss = ((loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()

                        # Compute prior loss
                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                        # Add the prior loss to the instance loss.
                        loss = loss + args.prior_loss_weight * prior_loss
                    else:
                        mask = batch[mask_key] * (args.bg_loss_weight*(1-batch[data_mask_key]) + batch[data_mask_key])
                        # print("min:", mask.min())
                        # print("max:", mask.max())
                        # print("mean:", mask.mean())
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        loss = ((loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()
                
                    
                accelerator.backward(loss)
                # Zero out the gradients for all token embeddings except the newly added
                # embeddings for the concept, as we only want to optimize the concept embeddings
                if args.modifier_token is not None:
                    if accelerator.num_processes > 1:
                        grads_text_encoder = text_encoder.module.get_input_embeddings().weight.grad
                    else:
                        grads_text_encoder = text_encoder.get_input_embeddings().weight.grad
                    # Get the index for tokens that we want to zero the grads for
                    index_grads_to_zero = torch.arange(len(tokenizer)) != modifier_token_id[0]
                    for i in range(len(modifier_token_id[1:])):
                        index_grads_to_zero = index_grads_to_zero & (
                            torch.arange(len(tokenizer)) != modifier_token_id[i]
                        )
                    grads_text_encoder.data[index_grads_to_zero, :] = grads_text_encoder.data[
                        index_grads_to_zero, :
                    ].fill_(0)

                if accelerator.sync_gradients:
                    if args.cd_only_on_controlnet:
                        params_to_clip = (
                            itertools.chain(text_encoder.parameters(), op_custom_diffusion_layers.parameters())
                            if args.modifier_token is not None
                            else op_custom_diffusion_layers.parameters()
                        )
                    else:
                        params_to_clip = (
                            itertools.chain(text_encoder.parameters(), custom_diffusion_layers.parameters())
                            if args.modifier_token is not None
                            else custom_diffusion_layers.parameters()
                        )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            if accelerator.is_main_process:
                images = []

                #### Save actually queried images
                if args.save_intermediate_for_debug:
                    if global_step % 10 == 0:
                        b_size = len(batch['pixel_values'])
                        b_size = b_size // 2 if (args.with_prior_preservation and not args.fullbody_prior) else b_size
                        # b_size = b_size + 1 if (args.fullbody_prior) else b_size

                        input_img = (batch['pixel_values'] + 1)*255/2
                        cond_img = (batch['cond_imgs'])*255
                        data_mask = (batch['data_mask'] + (1-batch['data_mask'])*args.bg_loss_weight) * 255
                        data_mask = F.interpolate(data_mask, (input_img.shape[-2], input_img.shape[-1]), mode="bilinear", align_corners=False)
                        model_mask = batch['mask'] * 255
                        model_mask = F.interpolate(model_mask, (input_img.shape[-2], input_img.shape[-1]), mode="bilinear", align_corners=False)
                        prompts = batch['raw_prompts']

                        input_img = input_img.detach().cpu().permute(0,2,3,1)
                        cond_img = cond_img.detach().cpu().permute(0,2,3,1)
                        model_mask = model_mask.detach().cpu().permute(0,2,3,1)
                        data_mask = data_mask.detach().cpu().permute(0,2,3,1)

                        if args.with_prior_preservation and (not args.fullbody_prior):
                            p_input_img = input_img[b_size:]
                            p_cond_img = cond_img[b_size:] if len(cond_img) > b_size else torch.zeros_like(p_input_img)
                            p_model_mask = model_mask[b_size:]
                            
                            input_img = input_img[:b_size]
                            cond_img = cond_img[:b_size]
                            model_mask = model_mask[:b_size]
                            prompt = prompts[:b_size]

                            p_input_img = torch.cat([p_i for p_i in p_input_img], dim=0).numpy().astype(np.uint8)[..., ::-1]     
                            _p_cond_img = []
                            for p_i, _prompt in zip(p_cond_img, prompt):
                                p_i = p_i.numpy().astype(np.uint8)[..., ::-1]
                                p_i = img_add_text(p_i.copy(), f"openpose, priors")
                                _p_cond_img.append(p_i)
                            p_cond_img = np.concatenate(_p_cond_img, axis=0)
                            p_model_mask = torch.cat([p_i for p_i in p_model_mask], dim=0).repeat(1,1,3).numpy().astype(np.uint8)

                            p_input_img = img_add_text(p_input_img, f"gt")
                            p_model_mask = img_add_text(p_model_mask, f"model mask")     

                            p_img = np.concatenate([p_input_img, p_cond_img, p_model_mask], axis=1)
                            p_img = np.concatenate([
                                np.ones((200, p_img.shape[1], 3), dtype=np.uint8),
                                p_img
                            ], axis=0)
                            p_img = img_add_text(p_img, f"prior image, {global_step:06}")
                            (Path(args.output_dir) / 'debug_logs').mkdir(exist_ok=True)
                            save_fname = Path(args.output_dir) / 'debug_logs' / f"prior_{global_step:06}.jpg"
                            cv2.imwrite(str(save_fname), p_img)
                        

                        input_img = torch.cat([p_i for p_i in input_img], dim=0).numpy().astype(np.uint8)[..., ::-1]
                        _cond_img = []
                        for p_i, _prompt in zip(cond_img, prompts):
                            p_i = p_i.numpy().astype(np.uint8)[..., ::-1]
                            p_i = img_add_text(p_i.copy(), f"openpose, prompt: {str(_prompt)}")
                            _cond_img.append(p_i)
                        cond_img = np.concatenate(_cond_img, axis=0)
                        model_mask = torch.cat([p_i for p_i in model_mask], dim=0).repeat(1,1,3).numpy().astype(np.uint8)[..., ::-1]
                        data_mask = torch.cat([p_i for p_i in data_mask], dim=0).repeat(1,1,3).numpy().astype(np.uint8)[..., ::-1]

                        input_img = img_add_text(input_img, f"gt")
                        model_mask = img_add_text(model_mask, f"model mask") 
                        data_mask = img_add_text(data_mask, f"data mask")    

                        log_img = np.concatenate([input_img, cond_img, model_mask, data_mask], axis=1)
                        log_img = np.concatenate([
                            np.ones((200, log_img.shape[1], 3), dtype=np.uint8),
                            log_img
                        ], axis=0)
                        log_img = img_add_text(log_img, f"main image, {global_step:06}")
                        (Path(args.output_dir) / 'debug_logs').mkdir(exist_ok=True)
                        save_fname = Path(args.output_dir) / 'debug_logs' / f"main_{global_step:06}.jpg"
                        cv2.imwrite(str(save_fname), log_img)


                ### Validation every args.validation_steps
                if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                    with torch.no_grad():
                        logger.info(
                            f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                            f" {args.validation_prompt}."
                        )
                        if controlnet_op is None:
                            from diffusers import ControlNetModel
                            controlnet_op = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=weight_dtype)
                            controlnet_op.to(accelerator.device)

                        
                        # create pipeline
                        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet),
                            text_encoder=accelerator.unwrap_model(text_encoder),
                            tokenizer=tokenizer,
                            revision=args.revision,
                            safety_checker=None,
                            controlnet=controlnet_op, 
                            torch_dtype=weight_dtype,
                        )
                        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
                        pipeline = pipeline.to(accelerator.device)
                        pipeline.set_progress_bar_config(disable=True)

                        canon_op_img = Image.open(Path(__file__).parent / "canon_openpose.png")

                        # run inference
                        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                        if args.cd_only_on_controlnet:
                            images = [
                                pipeline(args.validation_prompt, args.instance_prompt_wo_token, num_inference_steps=25, generator=generator, eta=1.0, image=canon_op_img).images[
                                    0
                                ]
                                for _ in range(args.num_validation_images)
                            ]
                        else:
                            images = [
                                pipeline(args.validation_prompt, num_inference_steps=25, generator=generator, eta=1.0, image=canon_op_img).images[
                                    0
                                ]
                                for _ in range(args.num_validation_images)
                            ]

                        for tracker in accelerator.trackers:
                            if tracker.name == "tensorboard":
                                np_images = np.stack([np.asarray(img) for img in images])
                                tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                            if tracker.name == "wandb":
                                tracker.log(
                                    {
                                        "validation": [
                                            wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                            for i, image in enumerate(images)
                                        ]
                                    }
                                )
                            
                            # Also save validation images, to make a video
                            for i, image in enumerate(images):
                                (Path(args.output_dir) / 'optim_logs').mkdir(exist_ok=True)
                                save_fname = Path(args.output_dir) / 'optim_logs' / f"{i}_{global_step:06}.jpg"
                                img = np.asarray(image)[..., ::-1]

                                img = img_add_text(img, f"{global_step:06}")
                                cv2.imwrite(str(save_fname), img)


                        del pipeline
                        torch.cuda.empty_cache()

    # Save the custom diffusion layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.cd_only_on_controlnet:
            controlnet_op = controlnet_op.to(torch.float32)
            save_attn_procs(controlnet_op, args.output_dir, safe_serialization=not args.no_safe_serialization)
        else:
            unet = unet.to(torch.float32)
            unet.save_attn_procs(args.output_dir, safe_serialization=not args.no_safe_serialization)
        save_new_embed(
            text_encoder,
            modifier_token_id,
            accelerator,
            args,
            args.output_dir,
            safe_serialization=not args.no_safe_serialization,
        )
        
        if global_step % 1000 == 0 and global_step > 0:
            epoch_save_dir = f"{args.output_dir}/token_{global_step}"
            os.makedirs(epoch_save_dir, exist_ok=True)

            if args.cd_only_on_controlnet:
                controlnet_op = controlnet_op.to(torch.float32)
                save_attn_procs(controlnet_op, epoch_save_dir, safe_serialization=not args.no_safe_serialization)
            else:
                unet = unet.to(torch.float32)
                unet.save_attn_procs(epoch_save_dir, safe_serialization=not args.no_safe_serialization)


            save_new_embed(
                text_encoder,
                modifier_token_id,
                accelerator,
                args,
                epoch_save_dir,
                safe_serialization=not args.no_safe_serialization,
            )
 
        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                prompt=args.instance_prompt,
                repo_folder=args.output_dir,
            )
            api = HfApi(token=args.hub_token)
            api.upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()