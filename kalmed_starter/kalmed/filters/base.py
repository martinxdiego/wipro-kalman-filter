from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class FilterState:
    x: np.ndarray  # state vector (n,)
    P: np.ndarray  # covariance (n,n)

class AbstractFilter:
    def predict(self, *args, **kwargs):  # pragma: no cover - interface
        raise NotImplementedError
    def update(self, *args, **kwargs):   # pragma: no cover - interface
        raise NotImplementedError
