from typing import Tuple

import torch
from jaxtyping import Float
from torch import Tensor


def coeffs_to_sine(
    a: Float[Tensor, "..."], b: Float[Tensor, "..."], x: Float[Tensor, "..."]
) -> Float[Tensor, "..."]:
    """
    Computes a linear combination of cosine and sine functions given coefficients a and b.

    The function returns the value a*cos(x) + b*sin(x) for each element in x.

    Args:
        a: Coefficient for the cosine term.
        b: Coefficient for the sine term.
        x: Tensor of angles.

    Returns:
        A tensor of the same shape as x representing a*cos(x) + b*sin(x).
    """
    raise NotImplementedError("Homework!")


def angle_to_coeffs(
    angle: Float[Tensor, "..."],
) -> Tuple[Float[Tensor, "..."], Float[Tensor, "..."]]:
    """
    Converts an angle to its corresponding cosine and sine coefficients.

    This function computes the cosine and sine of the provided angle, effectively mapping the angle
    to a point on the unit circle.

    Args:
        angle: Tensor of angles.

    Returns:
        A tuple (cos(angle), sin(angle)) of tensors, each with the same shape as the input.
    """
    raise NotImplementedError("Homework!")
