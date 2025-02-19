from typing import Callable

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.data import astronaut as _astronaut
from skimage.data import hubble_deep_field

from .gauss import _gaussian_filter_1d


def gaussian_filter(
    img: torch.Tensor,  # The input tensor
    sigma: float,  # Standard deviation for the Gaussian kernel
    order: int | list = 0,  # The order of the filter's derivative along each dim
    mode: str = "reflect",  # Padding mode for `torch.nn.functional.pad`
    truncate: float = 4.0,  # Number of standard deviations to sample the filter
) -> torch.Tensor:
    """
    Convolves an image with a Gaussian kernel (or its derivatives).
    """

    # Specify the dimensions of the convolution to use
    ndim = img.ndim - 2
    if isinstance(order, int):
        order = [order] * ndim
    else:
        assert len(order) == ndim, "Specify the Gaussian derivative order for each dim"
    convfn = getattr(F, f"conv{ndim}d")

    # Convolve along the rows, columns, and depth (optional)
    for dim, derivative_order in enumerate(order):
        img = _conv(img, convfn, sigma, derivative_order, truncate, mode, dim)
    return img


def _conv(
    img: torch.Tensor,
    convfn: Callable,
    sigma: float,
    order: int,
    truncate: float,
    mode: str,
    dim: int,
):
    # Make a 1D kernel and pad such that the image size remains the same
    kernel = _gaussian_filter_1d(sigma, order, truncate, img.dtype, img.device)
    padding = len(kernel) // 2

    # Specify the padding dimensions
    pad = [0] * 2 * (img.ndim - 2)
    for idx in range(2 * dim, 2 * dim + 2):
        pad[idx] = padding
    pad = pad[::-1]
    x = F.pad(img, pad, mode=mode)

    # Specify the dimension along which to do the convolution
    view = [1] * img.ndim
    view[dim + 2] *= -1

    return convfn(x, weight=kernel.view(*view))


def astronaut(dtype: torch.dtype = torch.float32):
    img = _astronaut()
    img = img_as_float(img)
    img = rgb2gray(img)
    img = torch.from_numpy(img).to(dtype)
    return img[None, None]


def hubble(dtype: torch.dtype = torch.float32):
    img = hubble_deep_field()[0:500, 0:500]
    img = img_as_float(img)
    img = rgb2gray(img)
    img = torch.from_numpy(img).to(dtype)
    return img[None, None]


def imshow(*imgs):
    imgs = [img.squeeze().detach().cpu() for img in imgs]
    n_imgs = len(imgs)
    fig, axs = plt.subplots(ncols=n_imgs, figsize=(n_imgs * 2 + 2, 3))
    for ax, img in zip(axs, imgs):
        ax.imshow(img, cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    return fig, axs
