import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import typing as tp
from torchvision.utils import make_grid
from torchvision.transforms import Resize
import torchvision.transforms as transforms
from PIL import Image


def get_image_t(img_path, size=1024):
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.Resize(size)
    ])
    img = Image.open(str(img_path)).convert('RGB')
    return image_transform(img).unsqueeze(0)


def t2im(img_t: torch.Tensor, size: int = 512):
    """
    process torch image with shape (3, h, w) to numpy image

    Parameters
    ----------
    img_t : torch.Tensor
        Image with shape (3, H, W)

    size : int
        Size for which smaller edge will be resized

    Returns
    -------
    img : np.ndarray
        Image with shape (3, H, W) with smaller edge resized to parameter 'size'
    """
    img = Resize(size)(img_t).permute(1, 2, 0).cpu().detach().numpy()
    img = np.round((np.clip(img, -1, 1) + 1) / 2 * 255).astype(np.uint8)
    return img


def resize_batch(imgs: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(imgs, (size, size))


def resize_img(img: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(img.unsqueeze(0), (size, size))[0]


@torch.no_grad()
def construct_paper_image_grid(img: torch.Tensor):
    """
    process torch batch image to paper image

    Parameters
    ----------
    img : torch.Tensor
        Image batch with shape (16, 3, H, W)

    Returns
    -------
    base_fig : np.ndarray
        Image with shape (3, H, W) with smaller edge resized to 512
    """
    half_size = img.size()[-1] // 2
    quarter_size = half_size // 2

    base_fig = torch.cat([img[0], img[1]], dim=2)
    sub_cols = [torch.cat([resize_img(img[i + j], half_size) for j in range(2)], dim=1) for i in range(2, 8, 2)]
    base_fig = torch.cat([base_fig, *sub_cols], dim=2)

    sub_cols = [torch.cat([resize_img(img[i + j], quarter_size) for j in range(4)], dim=1) for i in range(8, 16, 4)]
    base_fig = torch.cat([base_fig, *sub_cols], dim=2)

    base_fig = Resize(512)(base_fig).permute(1, 2, 0).cpu().detach().numpy()
    base_fig = np.round((np.clip(base_fig, -1, 1) + 1) / 2 * 255).astype(np.uint8)
    return base_fig


def construct_image_grid(imgs_t: tp.List[torch.Tensor], header=None, size=256, index: tp.List[torch.Tensor] = None):
    nrow = imgs_t[0].shape[0]
    if index is not None:
        nrow += 1
    res = torch.zeros(0, 3, size, size)

    if header is not None:
        res = torch.zeros(1, 3, size, size)
        res = torch.cat([res, resize_batch(header, size)])

    for i in range(len(imgs_t)):
        row = resize_batch(imgs_t[i], size)
        if index is not None:
            row = torch.cat([resize_batch(index[i], size), row])
        res = torch.cat([res, row])
    grid = make_grid(res, nrow=nrow)
    return t2im(grid, size=grid.shape[1:])


def crop_augmentation(image: torch.Tensor, size=1024, alpha=0.8):
    max_ = int(size * (1 - alpha))
    len_ = int(size * alpha)
    x, y = np.random.randint(max_, size=2)
    return image[..., x:x + len_, y:y + len_]
