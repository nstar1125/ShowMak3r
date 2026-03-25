
import torch
import torch.nn as nn
import lpips
import json

import numpy as np
from torch.cuda.amp import custom_fwd
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def convert_to_np(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    
    elif isinstance(input, np.ndarray):
        return input
    
    else:
        return np.array(input)


def convert_tensor_to_number(data):
    if isinstance(data, dict):
        return {key: convert_tensor_to_number(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_tensor_to_number(item) for item in data]
    elif isinstance(data, torch.Tensor):
        return data.item()  # Extract numerical value from the tensor
    elif isinstance(data, type(None)):
        return dict()  # if it's None. convert to -1
    else:
        return data

def scale_for_lpips(image_tensor):
    return image_tensor * 2. - 1.


class Evaluator(nn.Module):
    """ https://github.com/JanaldoChen/Anim-NeRF/blob/main/models/evaluator.py"""
    def __init__(self):
        super().__init__()
        
        self.lpips_alex = lpips.LPIPS(net='alex', version='0.1') # best forward scores
        # loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization


        # self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex")
        self.ssim_kernel_size = 11
        self.psnr = PeakSignalNoiseRatio(data_range=1)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1, kernel_size=self.ssim_kernel_size)
        self.raw_ssim = StructuralSimilarityIndexMeasure(data_range=1, reduction='none', kernel_size=self.ssim_kernel_size, return_full_image=True)

    # custom_fwd: turn off mixed precision to avoid numerical instability during evaluation
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, rgb, rgb_gt, mask=None, mask_gt=None, get_both=True):
        """
        Assume 0~1 normalized output, [N, H, W, C] inputs
        """
        self.raw_ssim.reset()
        self.psnr.reset()
        self.ssim.reset()

        # torchmetrics assumes NCHW format
        if len(rgb.shape) == 3:
            rgb = rgb[None]
        if len(rgb_gt.shape) == 3:
            rgb_gt = rgb_gt[None]
        if mask is not None and len(mask.shape) == 3 and (mask.shape[-1]==1):
            mask = mask[None]
        elif mask is not None and len(mask.shape) == 2:
            mask = mask[None][...,None]
        if mask_gt is not None and len(mask_gt.shape) == 3 and (mask_gt.shape[-1]==1):
            mask_gt = mask_gt[None]
        elif mask_gt is not None and len(mask_gt.shape) == 2:
            mask_gt = mask_gt[None][...,None]

        get_masked = False
        if mask is not None:
            mask = mask.permute(0, 3, 1, 2)     # NCHW format
        if mask_gt is not None:
            mask_gt = mask_gt.permute(0, 3, 1, 2)
            get_masked = True

        # rgb = rgb.permute(0, 3, 1, 2).clamp(max=1.0)
        # rgb_gt = rgb_gt.permute(0, 3, 1, 2)

        if get_both and get_masked:
            return {
                "psnr": self.psnr(rgb, rgb_gt).mean().detach().cpu(),
                "ssim": self.ssim(rgb, rgb_gt).mean().detach().cpu(),
                "gt_mask_psnr": self.masked_metric(rgb, rgb_gt, mask_gt, 'psnr').detach().cpu(),
                "gt_mask_ssim": self.masked_metric(rgb, rgb_gt, mask_gt, 'ssim').detach().cpu(),
                "lpips": self.lpips(rgb, rgb_gt).detach().cpu(),
                "gt_mask_lpips": self.masked_lpips(rgb, rgb_gt, mask_gt).detach().cpu(),
                "mask_iou": self.mask_iou(mask, mask_gt).detach().cpu()
            }
        
        elif get_masked:
            return {
                "gt_mask_psnr": self.masked_metric(rgb, rgb_gt, mask_gt, 'psnr').detach().cpu(),
                "gt_mask_ssim": self.masked_metric(rgb, rgb_gt, mask_gt, 'ssim').detach().cpu(),
                "gt_mask_lpips": self.masked_lpips(rgb, rgb_gt, mask_gt).detach().cpu(),
                "mask_iou": self.mask_iou(mask, mask_gt).detach().cpu()
            }
        else:
            return {
                "psnr": self.psnr(rgb, rgb_gt).mean().detach().cpu(),
                "ssim": self.ssim(rgb, rgb_gt).mean().detach().cpu(),
                "lpips": self.lpips(rgb, rgb_gt).detach().cpu(),
            }
    

    def lpips(self, rgb, rgb_gt):
        return self.lpips_alex(scale_for_lpips(rgb), scale_for_lpips(rgb_gt))
    

    def masked_lpips(self, rgb, rgb_gt, mask):
        mask = mask > 0.5
        mask = mask.repeat(1,3,1,1)
        rgb = scale_for_lpips(rgb)
        rgb[~mask] *= 0
        rgb_gt = scale_for_lpips(rgb_gt)
        rgb_gt[~mask] *= 0
        return self.lpips_alex(rgb, rgb_gt)
    

    def mask_iou(self, mask, mask_gt):
        # Convert masks to binary arrays
        mask = convert_to_np(mask)  > 0.5  # Assuming pixel values above 0.5 are considered foreground
        mask_gt = convert_to_np(mask_gt) > 0.5

        # Calculate intersection and union
        intersection = np.logical_and(mask, mask_gt)
        union = np.logical_or(mask, mask_gt)

        # Compute IoU
        iou = np.sum(intersection) / np.sum(union)
        iou = torch.tensor([iou])

        return iou
    

    def masked_metric(self, rgb, rgb_gt, mask, mode='ssim'):
        mask = mask > 0.5
        rgb = rgb * mask 
        rgb_gt = rgb_gt * mask

        if mode == 'ssim':
            _, metric = self.raw_ssim(rgb, rgb_gt)
            
            pad_size = (self.ssim_kernel_size-1)//2
            metric = metric[:,:,pad_size:-pad_size, pad_size:-pad_size]
            ssim_mask = mask[:,:,pad_size:-pad_size, pad_size:-pad_size]
            masked_ssim = (metric * ssim_mask).sum() / (ssim_mask.sum() * 3)
            mean_metric = masked_ssim
        elif mode == 'psnr':
            self.psnr.reset()
            psnr = self.psnr(rgb, rgb_gt).mean()
            mean_metric = psnr + (10 / torch.log(torch.tensor(10))) * (torch.log(mask.sum()*3) - torch.log(torch.tensor(rgb.numel(), device=psnr.device)))
        else:
            raise AssertionError()

        
        return mean_metric
        

    def dump_as_json(self, save_fname, data_dict):
        data_dict = convert_tensor_to_number(data_dict)
        with open(save_fname, 'w') as fp:
            json.dump(data_dict, fp, indent=True)
