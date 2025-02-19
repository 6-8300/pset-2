import torch
import torch.nn.functional as F
from jaxtyping import Float, Complex
from torch import Tensor

from .gauss import _gaussian_filter_1d


def oriented_filter(theta: float, sigma: float, **kwargs) -> Float[Tensor, "N N"]:
    """
    Return an oriented first-order Gaussian filter
    given an angle (in radians) and standard deviation.

    Hint:
    - Use `.gauss._gaussian_filter_1d`!

    Implementation details:
    - **kwargs are passed to `_gaussian_filter_1d`
    """
    raise NotImplementedError("Homework!")


def conv(
    img: Float[Tensor, "B 1 H W"],  # Input image
    kernel: Float[Tensor, "N N"] | Complex[Tensor, "N N"],  # Convolutional kernel
    mode: str = "reflect",  # Padding mode
) -> Float[Tensor, "B 1 H W"]:
    """
    Convolve an image with an oriented first-order Gaussian filter
    given an angle (in radians) and standard deviation.
    """
    raise NotImplementedError("Homework!")


def steer_the_filter(
    img: Float[Tensor, "B 1 H W"], theta: float, sigma: float, **kwargs
) -> Float[Tensor, "B 1 H W"]:
    """
    Return the image convolved with a steered filter.
    """
    raise NotImplementedError("Homework!")


def steer_the_images(
    img: Float[Tensor, "B 1 H W"], theta: float, sigma: float, **kwargs
) -> Float[Tensor, "B 1 H W"]:
    """
    Return the steered image convolved with a filter.
    """
    raise NotImplementedError("Homework!")


def measure_orientation(
    img: Float[Tensor, "B 1 H W"], sigma: float, **kwargs
) -> Float[Tensor, "B 1 H W"]:
    """
    Design a filter to measure the orientation of edges in an image.

    Hint:
    - Consider the complex filter from the README
    - You will need to design a method for noise suppression
    """
    raise NotImplementedError("Homework!")
