from typing import Tuple
import cv2 as cv
import imagesize


def read_bgr_image(path: str):
    return cv.imread(path)


def read_rgb_image(path: str):
    return cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)


def read_image_shape(path: str) -> Tuple[int, int]:
    return imagesize.get(path)  # type: ignore

def read_image_from_tensor(tensor):
    return tensor.numpy().transpose(1, 2, 0) * 255



def write_bgr_image(path: str, image: cv.Mat):
    return cv.imwrite(path, image)


def write_rgb_image(path: str, image: cv.Mat):
    return cv.imwrite(path, cv.cvtColor(image, cv.COLOR_RGB2BGR))


def write_greyscale_image(path: str, image: cv.Mat):
    return cv.imwrite(path, image)
