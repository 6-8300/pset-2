from jaxtyping import Float
from torch import Tensor

from .provided import gaussian_filter


def zeroth_order(img: Float[Tensor, "*H W"], sigma: float) -> Float[Tensor, "*H W"]:
    raise NotImplementedError("Homework!")


def first_order_x(img: Float[Tensor, "*H W"], sigma: float) -> Float[Tensor, "*H W"]:
    raise NotImplementedError("Homework!")


def first_order_y(img: Float[Tensor, "*H W"], sigma: float) -> Float[Tensor, "*H W"]:
    raise NotImplementedError("Homework!")


def first_order_xy(img: Float[Tensor, "*H W"], sigma: float) -> Float[Tensor, "*H W"]:
    raise NotImplementedError("Homework!")


def second_order_xx(img: Float[Tensor, "*H W"], sigma: float) -> Float[Tensor, "*H W"]:
    raise NotImplementedError("Homework!")


def second_order_yy(img: Float[Tensor, "*H W"], sigma: float) -> Float[Tensor, "*H W"]:
    raise NotImplementedError("Homework!")


def log(img: Float[Tensor, "*H W"], sigma: float) -> Float[Tensor, "*H W"]:
    raise NotImplementedError("Homework!")
