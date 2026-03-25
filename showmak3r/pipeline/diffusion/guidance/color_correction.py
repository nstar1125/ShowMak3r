import cv2
import numpy as np

def compute_cc_target(target_images):
    """
    Calculate the color histogram (LAB format) given images. 
    (to fit the histogram of two images)
    """
    if target_images is None or len(target_images)==0:
        return None

    target_histogram = (cv2.cvtColor(np.asarray(target_images[0].copy()), cv2.COLOR_RGB2LAB)*0).astype('float64')
    for img in target_images:
        target_histogram_component = cv2.cvtColor(np.asarray(img.copy()), cv2.COLOR_RGB2LAB).astype('float64')
        target_histogram += (target_histogram_component/len(target_images)).astype('float64')
                
    target_histogram=target_histogram.astype('uint8')
    
    return target_histogram

def apply_color_correction(target_histogram, original_image):
    """
    Shift the histograms (LAB) of given image to given target_histogram. 
    (to fit the histogram of two images)
    """
    image = cv2.cvtColor(exposure.match_histograms(
        cv2.cvtColor(
            np.asarray(original_image),
            cv2.COLOR_RGB2LAB
        ),
        correction,
        channel_axis=2
    ), cv2.COLOR_LAB2RGB).astype("uint8")

    return image

