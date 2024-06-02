import torch as th
from typing import Optional


def save_state(checkpoint_path: str,
               model: th.nn.Module,
               optimizer: Optional[th.optim.Optimizer] = None,
               scheduler: Optional[th.optim.lr_scheduler.LRScheduler] = None) -> None:
    """
    Save the model, optimizer and scheduler state dicts.
    """

    state_dict = {
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        state_dict["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state_dict["scheduler_state_dict"] = scheduler.state_dict()
    th.save(state_dict, checkpoint_path)


def load_state(checkpoint_path: str,
               model: Optional[th.nn.Module] = None,
               optimizer: Optional[th.optim.Optimizer] = None,
               scheduler: Optional[th.optim.lr_scheduler.LRScheduler] = None) -> None:
    """
    Load the model, optimizer and scheduler state dicts.
    """

    checkpoint = th.load(checkpoint_path)
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


def count_parameters(model: th.nn.Module) -> int:
    """
    Count the number of parameters in a model.
    """

    return sum(p.numel() for p in model.parameters())


class _round(th.autograd.Function):
    """
    Round autograd function.
    """

    @staticmethod
    def forward(ctx, inpt: th.Tensor) -> th.Tensor:
        return inpt.round()

    @staticmethod
    def backward(ctx, grad: th.Tensor) -> th.Tensor:
        return grad


def quantize(inpt: th.Tensor) -> th.Tensor:
    """
    Differentiable quantization.
    """

    return _round.apply(255. * inpt.clamp(0., 1.)) / 255.


def rgb2y(inpt: th.Tensor) -> th.Tensor:
    """
    Convert RGB to Y(CbCr).
    """

    return (65.481 * inpt[:, :1, ...] + 128.553 * inpt[:, 1:2, ...] + 24.966 * inpt[:, 2:, ...] + 16.) / 255.


def charbonnier_loss(x: th.Tensor, y: th.Tensor, eps: float = 1e-3) -> th.Tensor:
    """
    Charbonnier loss.
    """

    return ((x - y) ** 2 + eps ** 2).sqrt().sum() / x.shape[0]


def modcrop(x: th.Tensor, scale: int) -> th.Tensor:
    """
    Crop the input image dimensions to be a multiple of the scale.
    """

    h = x.shape[-2] - x.shape[-2] % scale
    w = x.shape[-1] - x.shape[-1] % scale
    return x[..., :h, :w]
