# %%
from dataclasses import dataclass
from typing import Tuple, Callable


@dataclass
class BathMode:
    """Extended definition of the bath modes
        z_j equiv q_j if sig = 1
        z_j equiv p_j if sig = -1
    Extended definition of the bath modes:
        BathMode(sig, n) = \sum_j g_j \omega_j^n z_j(sig)
    where g_j characterizes the spectral density and \omega_j
    is the frequency of the j-th mode.
    """
    sig: int # signature = (1, -1), 1 for position, -1 for momentum
    n: int   # power of omega

    def __str__(self):
        if self.sig == 1:
            sig = "ρ"
        else:
            sig = "π"
        return f"{sig}_{self.n}"

def BathModeCommute(
    mode1: BathMode,
    mode2: BathMode,
    theta_func: Callable[[int], complex]
) -> complex:
    if mode1.sig == mode2.sig:
        return 0.0
    else:
        # \sum_j \sum_k g_j g_k \omega_j^n \omega_k^m [z_j, z_k]
        sign = 1.j if mode1.sig == 1 else -1.j
        n = mode1.n + mode2.n
        return sign * theta_func(n)

def apply_iLB(mode: BathMode) -> Tuple[int, BathMode]:
    # apply i [H_B, BathMode(sig, n)]
    if mode.sig == 1:
        return 1, BathMode(sig=-1, n=mode.n+1)
    else:
        return -1, BathMode(sig=1, n=mode.n+1)