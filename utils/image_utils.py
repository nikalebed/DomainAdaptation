import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torchvision.transforms import Resize

def t2im(img_t: torch.Tensor, size: int = 512):
    """
    process torch image with shape (3, h, w) to numpy image

    Parameters
    ----------
    img_t : torch.Tensor
        Image batch with shape (16, 3, H, W)

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


def crop_augmentation(image: torch.Tensor, size=1024, alpha=0.8):
    max_ = int(size * (1 - alpha))
    len_ = int(size * alpha)
    x, y = np.random.randint(max_, size=2)
    return image[..., x:x + len_, y:y + len_]
