from typing import Tuple
import cv2 as cv
import imagesize

def write_bgr_image(path: str, image: cv.Mat):
    return cv.imwrite(path, image)

def write_rgb_image(path: str, image: cv.Mat):
    return cv.imwrite(path, cv.cvtColor(image, cv.COLOR_RGB2BGR))
