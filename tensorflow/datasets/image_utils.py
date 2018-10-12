import functools
from PIL import Image, ImageEnhance
import math
import numpy as np


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions[::-1], lambda x: x)


def random_subset(f_list, min_len=0, max_len=1):
    return np.random.choice(f_list, randint(min_len, max_len), replace=False)


def load_image(path):
    return Image.open(path)


def randint(a, b):
    return np.random.randint(a, b + 1, 1)[0]


def resize(img, min_px=256, max_px=286):
    w = randint(min_px, max_px)
    h = randint(min_px, max_px)
    return img.resize((w, h), Image.BILINEAR)


def resize_aspect(img, min_px=400, max_px=400):
    width = img.size[0]
    height = img.size[1]
    smallest = min(width, height)
    largest = max(width, height)
    k = 1
    if largest > max_px:
        k = max_px / float(largest)
        smallest *= k
        largest *= k
    if smallest < min_px:
        k *= min_px / float(smallest)
    size = int(math.ceil(width * k)), int(math.ceil(height * k))
    img = img.resize(size, Image.BILINEAR)
    return img


def resize_aspect_random(img, min_px=128, max_px=128):
    px = randint(min_px, max_px)
    return resize_aspect(img, px, px)


def crop_rect(img, crop_size=(96, 48), center=True):
    width, height = img.size
    if not center:
        h_off = randint(0, height - crop_size[1])
        w_off = randint(0, width - crop_size[0])
    else:
        h_off = (height - crop_size[1]) / 2
        w_off = (width - crop_size[0]) / 2
    return img.crop((w_off, h_off, w_off + crop_size[0], h_off + crop_size[1]))


def contrast(img, steps=None):
    if steps is None:
        return img
    idx = randint(0, len(steps) - 1)
    enh = ImageEnhance.Contrast(img)
    return enh.enhance(steps[idx])


def brightness(img, steps=None):
    if steps is None:
        return img
    idx = randint(0, len(steps) - 1)
    enh = ImageEnhance.Brightness(img)
    return enh.enhance(steps[idx])


def saturation(img, steps=None):
    if steps is None:
        steps = [0]
    idx = randint(0, len(steps) - 1)
    enh = ImageEnhance.Color(img)
    return enh.enhance(steps[idx])


def get_color_jitter(min_v, max_v):
    contrast_f = functools.partial(contrast, steps=np.arange(min_v, max_v, step=0.1).tolist())
    brightness_f = functools.partial(brightness, steps=np.arange(min_v, max_v, step=0.1).tolist())
    color_jitter = lambda x: compose(*random_subset([contrast_f, brightness_f], min_len=0, max_len=2))(x)
    return color_jitter


def img2array(img):
    a = np.array(img)
    if len(a.shape) == 2:
        a = a[:, :, np.newaxis]
    return a.astype(np.float32)


train_pipeline = compose(load_image,
                         functools.partial(resize_aspect_random, min_px=72, max_px=128),
                         functools.partial(crop_rect, crop_size=(96, 48)),
                         functools.partial(resize, min_px=32, max_px=32),
                         get_color_jitter(0.8, 1.0),
                         img2array)

eval_pipeline = compose(load_image,
                        functools.partial(resize, min_px=96, max_px=96),
                        functools.partial(crop_rect, crop_size=(96, 48)),
                        functools.partial(resize, min_px=32, max_px=32),
                        img2array)
