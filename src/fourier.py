from typing import Tuple, TypeVar

import torch
from jaxtyping import Float
from torch import Tensor

N = TypeVar("N")


def shift_operator(img_shape: Tuple[int, int], shift_x: int, shift_y: int) -> Tensor:
    """
    Constructs a 2D shift operator for an image with circular boundaries.

    Args:
        img_shape: Tuple[int, int]
            The (height, width) dimensions of the image.
        shift_x: int
            The number of pixels to shift horizontally.
        shift_y: int
            The number of pixels to shift vertically.

    Returns:
        Tensor of shape (h*w, h*w)
            A matrix that, when applied to a flattened image, shifts it by the specified amounts.
    """
    raise NotImplementedError("Homework!")


def matrix_from_convolution_kernel(
    kernel: Float[Tensor, "*"], n: int
) -> Float[Tensor, "n n"]:
    """
    Constructs a circulant matrix of size n x n from a 1D convolution kernel with periodic alignment.

    Args:
        kernel: Tensor
            A 1D convolution kernel.
        n: int
            The desired size of the circulant matrix.

    Returns:
        Tensor of shape (n, n)
            The circulant matrix representing the convolution with periodic boundary conditions.
    """
    raise NotImplementedError("Homework!")


def image_operator_from_sep_kernels(
    img_shape: Tuple[int, int],
    kernel_x: Float[Tensor, "*"],
    kernel_y: Float[Tensor, "*"],
) -> Float[Tensor, "N N"]:
    """
    Constructs a 2D convolution operator for an image by combining separable 1D kernels.

    Args:
        img_shape: Tuple[int, int]
            The (height, width) dimensions of the image.
        kernel_x: Tensor
            The 1D convolution kernel to be applied horizontally.
        kernel_y: Tensor
            The 1D convolution kernel to be applied vertically.

    Returns:
        Tensor of shape (h*w, h*w)
            The 2D convolution operator acting on a flattened image.
    """
    raise NotImplementedError("Homework!")


def eigendecomposition(
    operator: Float[Tensor, "N N"], descending: bool = True
) -> Tuple[Float[Tensor, "N"], Float[Tensor, "N N"]]:
    """
    Computes the eigenvalues and eigenvectors of a self-adjoint (Hermitian) linear operator.

    Args:
        operator: Tensor of shape (N, N)
            A self-adjoint linear operator.
        descending: bool
            If True, sort the eigenvalues and eigenvectors in descending order.

    Returns:
        A tuple (eigenvalues, eigenvectors) where:
            eigenvalues: Tensor of shape (N,)
            eigenvectors: Tensor of shape (N, N)
    """
    raise NotImplementedError("Homework!")


def fourier_transform_operator(
    operator: Float[Tensor, "N N"], basis: Float[Tensor, "N N"]
) -> Float[Tensor, "N N"]:
    """
    Computes the representation of a linear operator in the Fourier (eigen) basis.

    Args:
        operator: Tensor of shape (N, N)
            The original linear operator in pixel space.
        basis: Tensor of shape (N, N)
            The Fourier eigenbasis.

    Returns:
        Tensor of shape (N, N)
            The operator represented in the Fourier basis.
    """
    raise NotImplementedError("Homework!")


def fourier_transform(
    img: Float[Tensor, "N"], basis: Float[Tensor, "N N"]
) -> Float[Tensor, "N"]:
    """
    Projects a flattened image onto the Fourier (eigen) basis.

    Args:
        img: Tensor of shape (N,)
            A flattened image.
        basis: Tensor of shape (N, N)
            The Fourier eigenbasis.

    Returns:
        Tensor of shape (N,)
            The image represented in the Fourier domain.
    """
    raise NotImplementedError("Homework!")


def inv_fourier_transform(
    fourier_img: Float[Tensor, "N"], basis: Float[Tensor, "N N"]
) -> Float[Tensor, "N"]:
    """
    Reconstructs an image in pixel space from its Fourier coefficients using the provided eigenbasis.

    Args:
        fourier_img: Tensor of shape (N,)
            The image in the Fourier domain.
        basis: Tensor of shape (N, N)
            The Fourier eigenbasis used in the forward transform.

    Returns:
        Tensor of shape (N,)
            The reconstructed image in pixel space.
    """
    raise NotImplementedError("Homework!")
