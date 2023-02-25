import cv2 as cv
import imagesize
from typing import NamedTuple


def read_bgr_image(path: str):
    return cv.imread(path)


def read_rgb_image(path: str):
    return cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)


class ImageShape(NamedTuple):
    width: int
    height: int


def read_image_shape(path: str):
    width, height = imagesize.get(path)
    return ImageShape(int(width), int(height))


def read_image_from_tensor(tensor):
    return tensor.numpy().transpose(1, 2, 0) * 255


def write_bgr_image(path: str, image: cv.Mat):
    return cv.imwrite(path, image)


def write_rgb_image(path: str, image: cv.Mat):
    return cv.imwrite(path, cv.cvtColor(image, cv.COLOR_RGB2BGR))


def write_greyscale_image(path: str, image: cv.Mat):
    return cv.imwrite(path, image)
