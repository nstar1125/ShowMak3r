from math import floor
import cv2
def get_crop_img(img, bbox, rescale: float=1.2, resize: int=512, get_new_bbox=False):
    min_x = bbox[0]
    min_y = bbox[1]
    max_x = bbox[0] + bbox[2]
    max_y = bbox[1] + bbox[3]

    _w = int((max_x-min_x)*rescale) # size
    _h = int((max_y-min_y)*rescale)
    c_x = (min_x + max_x) // 2 # center
    c_y = (min_y + max_y) // 2

    w = _w if _w>_h else _h
    h = w

    x = floor(c_x - w//2) # left top corner
    y = floor(c_y - h//2)
    
    x_front, y_front, x_back, y_back = 0, 0, 0, 0

    # if the bbox is out of the image, apply padding
    if x < 0:
        x_front = -x
    if y < 0:
        y_front = -y
    if x+w >= img.shape[1]:
        x_back = x+w-img.shape[1]+1
    if y+h >= img.shape[0]:
        y_back = y+w-img.shape[0]+1

    # crop image
    if x_front+y_front+x_back+y_back > 0:
        ext_img = cv2.copyMakeBorder(img, y_front, y_back, x_front, x_back, cv2.BORDER_CONSTANT, value=(0,0,0))
        x = x + x_front
        y = y + y_front
    else:
        ext_img = img
    cropped_img = ext_img[y:y+h, x:x+h]

    if resize > 0:
        cropped_img = cv2.resize(cropped_img, (resize, resize))

    if get_new_bbox:
        if y_front > 0: # revert
            y = -y_front
        if x_front > 0:
            x = -x_front
        new_bbox = [x, y, w, h]
        
        return cropped_img, new_bbox
    else:
        return cropped_img


def calc_overlap(bbox_i, bbox_j, threshold: float=0.8)->bool:
    x1, y1, w1, h1 = bbox_i
    x2, y2, w2, h2 = bbox_j
    
    # Calculate coordinates of rectangles
    left1, right1 = x1, x1 + w1  
    top1, bottom1 = y1, y1 + h1
    left2, right2 = x2, x2 + w2
    top2, bottom2 = y2, y2 + h2
    
    # Calculate intersection area
    x_left = max(left1, left2)
    x_right = min(right1, right2)
    y_top = max(top1, top2)
    y_bottom = min(bottom1, bottom2)
    
    if x_right <= x_left or y_bottom <= y_top:
        return False
    
    # Calculate coverage
    intersection = (x_right - x_left) * (y_bottom - y_top)

    area1 = w1 * h1
    area2 = w2 * h2

    coverage = intersection / min(area1, area2)
    
    return coverage > threshold