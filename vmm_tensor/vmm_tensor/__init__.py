try:
    import torch  # noqa: F401
    from ._C import VMMTensor
except ImportError:
    VMMTensor = None

__all__ = ["VMMTensor"]
