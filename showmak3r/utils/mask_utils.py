import cv2
import scipy
import numpy as np
from numba import jit

@jit
def computeAlphaJit(alpha, unknown, b, h, w):
    alphaNew = alpha.copy()
    alphaOld = np.zeros(alphaNew.shape)
    threshold = 0.1
    n = 1
    while (n < 50 and np.sum(np.abs(alphaNew - alphaOld)) > threshold):
        alphaOld = alphaNew.copy()
        for i in range(1, h-1):
            for j in range(1, w-1):
                if(unknown[i,j]):
                    alphaNew[i,j] = 1/4  * (alphaNew[i-1 ,j] + alphaNew[i,j-1] + alphaOld[i, j+1] + alphaOld[i+1,j] - b[i,j])
        n +=1
    return alphaNew


def computeAlpha(alpha, unknown, b, h, w):
    """code from https://github.com/MarcoForte/poisson-matting/blob/master/Poisson%20Matting.ipynb"""
    alphaNew = alpha.copy()
    alphaOld = np.zeros(alphaNew.shape)
    threshold = 0.1
    n = 1
    ca = np.transpose(np.nonzero(unknown))
    c = ca[ (ca[:,0]>=1) & (ca[:,0]<h-1) &  (ca[:,1]>=1) & (ca[:,1]<w-1)]
    c0 = c[:,0]
    c1 = c[:,1]
    while (n < 50 and np.sum(np.abs(alphaNew - alphaOld)) > threshold):
        alphaOld = alphaNew.copy()
        alphaNew[c0, c1] = 1/4  * (alphaNew[c0 -1, c1] + alphaNew[c0, c1 -1] + alphaOld[c0,c1+1] + alphaOld[c0+1,c1] - b[c0,c1])
        n +=1
    return alphaNew


def matting_masks(img, mask, matting_pixels):
    matting_pixels = int(matting_pixels)
    # 1. First erode masks 
    true_region = erode_mask(mask.copy(), matting_pixels)
    bg_mask = (mask.max() - mask)
    true_bg = erode_mask(bg_mask.copy(), matting_pixels)
    fg = true_region == 255
    bg = true_bg == 255
    unknown = True ^ np.logical_or(fg,bg)
    
    # 2. Smooth F - B image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fg_img = gray_img*fg
    bg_img = gray_img*bg
    alphaEstimate = fg + 0.5 * unknown
    h, w = gray_img.shape
    
    approx_bg = cv2.inpaint(bg_img.astype(np.uint8),((unknown+fg)*255).astype(np.uint8),3,cv2.INPAINT_TELEA)*(np.logical_not(fg)).astype(np.float32)
    approx_fg = cv2.inpaint(fg_img.astype(np.uint8),((unknown+bg)*255).astype(np.uint8),3,cv2.INPAINT_TELEA)*(np.logical_not(bg)).astype(np.float32)
    approx_diff = approx_fg - approx_bg
    approx_diff = scipy.ndimage.filters.gaussian_filter(approx_diff, 0.9)
    approx_diff = approx_diff + 1e-9
    
    # 3. get gradient of image
    dy, dx = np.gradient(gray_img)
    d2y, _ = np.gradient(dy/approx_diff)
    _, d2x = np.gradient(dx/approx_diff)
    b = d2y + d2x
    
    # 4. get alpha
    alpha = computeAlphaJit(alphaEstimate, unknown, b, h, w)
    alpha = np.minimum(np.maximum(alpha,0),1).reshape(h,w)
    alpha = (alpha * 255).astype(np.uint8)
    
    # 5. visualize the results\
    input_viz = (alphaEstimate * 255).astype(np.uint8)
    viz = np.concatenate([alpha, input_viz, approx_bg.astype(np.uint8), approx_fg.astype(np.uint8), gray_img], axis=1)
    
    return alpha, viz
    



def dilate_mask(mask, kernel_size: int=30):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = mask.copy().astype(np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    return dilated_mask


def erode_mask(mask, erosion_size: int=5):
    # Create the kernel for erosion
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                        (erosion_size, erosion_size))
    # Perform erosion
    mask = mask.copy().astype(np.uint8)
    mask_eroded = cv2.erode(mask, element)

    return mask_eroded



def save_mask(fname, mask):
    if len(mask.shape) == 3:
        mask = mask.sum(-1)
    
    mask = mask > 0
    mask = (mask * 255).astype(np.uint8)

    cv2.imwrite(str(fname), mask)


def load_mask(fname):
    """
    It always return "discrete mask". (be aware of it)
    (it returns mask value range around 0~1)
    """
    mask = cv2.imread(str(fname), -1)

    if len(mask.shape) == 3:
        if mask.shape[-1] == 4:
            # RGBA case
            mask = mask[..., -1]
        
        if mask.shape[-1] == 3:
            mask = mask.sum(-1)
    
    if mask.max() > 0:
        mask = mask > (mask.max() * 0.5)
    mask = mask.astype(np.float32)
    
    return mask
        
    


