from math import factorial, ceil, sqrt

import torch
from jaxtyping import Float
from torch import Tensor


def _gaussian_filter_1d(
    sigma: float,  # Standard deviation of the Gaussian
    order: int,  # Order of the derivative
    truncate: float = 4.0,  # Truncate the filter at this many standard deviations
    dtype: torch.dtype = torch.float32,  # Data type to run the computation in
    device: torch.device = torch.device("cpu"),  # Device to run the computation on
) -> Float[Tensor, " filter_size"]:
    """
    Return a 1D Gaussian filter of a specified order.

    Implementation details:
    - filter_size = 2r + 1, where r = ceil(truncate * sigma)
    """
    raise NotImplementedError("Homework!")
