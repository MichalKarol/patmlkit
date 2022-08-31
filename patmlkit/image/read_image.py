from typing import Tuple
import cv2 as cv
import imagesize


def read_bgr_image(path: str):
    return cv.imread(path)


def read_rgb_image(path: str):
    return cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)


def read_image_shape(path: str) -> Tuple[int, int]:
    return imagesize.get(path)  # type: ignore
