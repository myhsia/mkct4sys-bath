# %%
import mpmath as mp
mp.dps = 31

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import yaml
from joblib import Memory


import os
import sys
import time
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Callable
from copy import deepcopy

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../library"))
from liouvillian.expval_bath_poly import expval_BathPoly
# from linear_moments import linear_moments
from liouvillian.poly_moments import poly_moments


cache_dir = "cache"
memory = Memory(location=cache_dir, verbose=0)
memory.clear(warn=False)

# load the parameters
with open("../params.yaml", "r") as f:
    params = yaml.safe_load(f)
    rescale = params["scale"]

# Define the parameters
Nmax_Omega = params["Pade_m"] + params["Pade_n"] + 2
Nmax_tilde_Omega = params["Nmax_tilde_Omega"]

lambd = params["lambd"]
wD = params["wD"]
Omega = params["Omega"]
Delta = params["Delta"]
beta = params["beta"]

# rescale the parameters to avoid numerical issues
beta /= rescale
wD, Omega, Delta= [
    x * rescale for x in [wD, Omega, Delta]]

sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])

Hs = Delta / 2 * sz + Omega / 2 * sx
V = sz / 2
mu = sx / 2

# linear coupling
# In the convention of this file, poly_coeffs are dimensionless
poly_coeffs = [1, 0]


# The initial system density matrix
# This basically assmues the initial time density matrix is rho = sigma_0 \otimes rho_B
# where sigma_0 is the initial system density matrix and rho_B is the bath density matrix
sigma_0 = np.diagflat([1+0.j, 0])

denominator = np.trace(mu @ mu @ sigma_0)


def J_mp(w):
    return 2 * lambd * w * mp.exp(-w / wD)


@memory.cache
def theta(n: int):
    def integrand(w):
        return 2 / mp.pi * J_mp(w) * w**n

    val_mp = mp.quad(integrand, [0, mp.inf])
    return float(val_mp)

@memory.cache
def eta(n: int):
    def integrand(w):
        return 1 / mp.pi * J_mp(w) * w**n / mp.tanh(beta * w / 2)

    val_mp = mp.quad(integrand, [0, mp.inf])
    return float(val_mp)

@memory.cache
def expval_BathPoly_wrapper(bp_str: str) -> complex:
    return expval_BathPoly(bp_str, theta, eta)


def main():
    parser = argparse.ArgumentParser(description="Parse command-line arguments for job settings.")

    # Define expected arguments
    parser.add_argument("-njobs", type=int, default=-1, help="Number of jobs to run in parallel")
    parser.add_argument("-inner_max_num_threads", type=int, default=1, help="Max number of threads per job")


    args = parser.parse_args()
    njobs = args.njobs
    inner_max_num_threads = args.inner_max_num_threads

    poly_moments(
        poly_coeffs,
        Nmax_Omega, Nmax_tilde_Omega,
        Hs, V, mu, sigma_0,
        theta, eta, expval_BathPoly_wrapper,
        njobs=njobs, innermax=inner_max_num_threads)

if __name__ == "__main__":
    main()

