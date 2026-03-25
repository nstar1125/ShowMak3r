import argparse
import os
import sys

import numpy as np
import json
import torch
from PIL import Image

import multiprocessing as mp
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))


# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

class ModelWrapper:
    def __init__(self, gpu_id, args):
        # cfg
        self.config_file = args.config  # change the path of the model config file
        self.grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
        self.sam_version = args.sam_version
        self.sam_checkpoint = args.sam_checkpoint
        self.sam_hq_checkpoint = args.sam_hq_checkpoint
        self.use_sam_hq = args.use_sam_hq
        self.input_dir = args.input_dir
        self.text_prompt = args.text_prompt
        self.output_dir = args.output_dir
        self.box_threshold = args.box_threshold
        self.text_threshold = args.text_threshold
        self.reverse_mask = args.reverse_mask
        self.device = f"cuda:{gpu_id}"
        torch.cuda.set_device(self.device)
        
        self.dino = self.load_model(self.config_file, self.grounded_checkpoint, device=self.device)
        if self.use_sam_hq:
            self.predictor = SamPredictor(
                sam_hq_model_registry[self.sam_version](checkpoint=self.sam_hq_checkpoint).to(self.device)
            )
        else:
            self.predictor = SamPredictor(
                sam_model_registry[self.sam_version](checkpoint=self.sam_checkpoint).to(self.device)
            )
        print(f"Loaded model on GPU {gpu_id}.")
    
    def process_image(self, t):
        image_pil, image = self.load_image(t)
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            return
        # run grounding dino model
        boxes_filt, pred_phrases = self.get_grounding_output(
            self.dino, image, self.text_prompt, self.box_threshold, self.text_threshold, device=self.device
        )
        image = cv2.imread(t)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image)
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)
        if(transformed_boxes.shape[0] > 0):
            masks, _, _ = self.predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes.to(self.device),
                multimask_output = False,
            )
        else:
            masks = None
        base = os.path.basename(t)
        base = os.path.splitext(base)[0]

        if masks != None:
            final_mask = masks.any(dim=0)        
            mask = final_mask.cpu().numpy()
            
            h, w = mask.shape[-2:]
            if self.reverse_mask: # white background, black mask 
                subtract_color = np.array([255, 255, 255, 0])
                main_color = np.array([255, 255, 255, 255])
                bg_image = np.full((h, w, 4), main_color, dtype=np.int64)
                mask_image = mask.reshape(h, w, 1) * subtract_color.reshape(1, 1, -1)
                final_image = bg_image.copy()
                final_image -= mask_image
            else: # black background, white mask
                subtract_color = np.array([255, 255, 255, 0])
                main_color = np.array([0, 0, 0, 255])
                bg_image = np.full((h, w, 4), main_color, dtype=np.int64)
                mask_image = mask.reshape(h, w, 1) * subtract_color.reshape(1, 1, -1)
                final_image = bg_image.copy()
                final_image += mask_image
        else:
            mask = None
            main_color = np.array([0, 0, 0, 255])
            bg_image = np.full((H, W, 4), main_color, dtype=np.int64)
            final_image = bg_image.copy()
        
        cv2.imwrite(f"{self.output_dir}/{base}.png", final_image, [cv2.IMWRITE_PNG_COMPRESSION, 0]) # save images 

    def load_image(self, image_path):
        # load image
        image_pil = Image.open(image_path).convert("RGB")  # load image

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image


    def load_model(self, model_config_path, model_checkpoint_path, device):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model


    def get_grounding_output(self, model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases

def process_batch(args, gpu_id, gpu_num, batch):
    print(f"\nProcessing batch on GPU {gpu_num}. batch size: {len(batch)}.")
    pipe = ModelWrapper(gpu_id, args)

    for t in tqdm(batch, desc=f"GPU {gpu_num}", total=len(batch)):
        pipe.process_image(t)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_dir", type=str, required=True, help="path to image directory")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--reverse_mask", action='store_true', help="reverse segmentation mask (default: black background, white mask)") # reverse mask
    parser.add_argument("--gpus", type=str, required=True, help="gpu ids")
    args = parser.parse_args()

    gpu_nums = args.gpus.split(',')
    gpu_size = len(gpu_nums)
    gpu_ids = [gid for gid in range(gpu_size)]
    ctx = mp.get_context('spawn')

    # make dir
    os.makedirs(args.output_dir, exist_ok=True)

    # fetch images
    if not os.path.isdir(args.input_dir):
        targets = [args.input_dir]
    else:
        targets = [
            f for f in os.listdir(args.input_dir) if not os.path.isdir(os.path.join(args.input_dir, f))
        ]
        targets = [os.path.join(args.input_dir, f) for f in targets]
    
    batch_size = len(targets) // gpu_size + (1 if len(targets) % gpu_size != 0 else 0)

    with ProcessPoolExecutor(max_workers=gpu_size, mp_context=ctx) as executor:
        process_list = []
        for gpu_id in gpu_ids:
            if gpu_id == gpu_size - 1: # last batch
                batch = targets[gpu_id * batch_size:]
            else: # other batches
                batch = targets[gpu_id * batch_size : (gpu_id + 1) * batch_size]
            process_arg = (
                args,
                gpu_id, 
                gpu_nums[gpu_id],
                batch
            )
            process_list.append(executor.submit(process_batch, *process_arg))
        for process in process_list:
            process.result() # wait for all process to complete
    
