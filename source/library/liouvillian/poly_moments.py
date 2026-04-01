import numpy as np
from liouvillian.bath_mode import BathMode
from liouvillian.bath_polynomial import BathPolynomial
from liouvillian.sb_general_term import SBGeneralTerm
from liouvillian.utils import tab_str
from liouvillian.apply_iLv import apply_iLv, apply_QiLv
from liouvillian.writer import MomentsWriter
import time
from typing import List, Tuple, Callable
from copy import deepcopy


def poly_moments(
    poly_coeffs: List[complex],
    Nmax_Omega: int,
    Nmax_tilde_Omega: int,
    Hs: np.ndarray,
    V: np.ndarray,
    mu: np.ndarray,
    sigma_0: np.ndarray,
    theta_func: Callable[[int], complex],
    eta_func: Callable[[int], complex],
    expval_func: Callable[[str], complex],
    njobs: int = -1,
    innermax: int = 1,
    quantum: bool = True
) -> None:
    """Compute the moments of polynomial bath interaction
        H = Hs + Hb + V (a_0 + a_1 x + a_2 x^2 + ... + a_n x^n)
    This script will return nothing but will always write the four files:
        - moments_quantum.dat: the moments Omega_n in quantum case
        - tilde_moments.dat: the moments tilde_Omega_n in quantum case
        - moments_classical.dat: the moments Omega_n in classical case
        - tilde_moments_classical.dat: the moments tilde_Omega_n in classical case

    Notice: will define Hs' = Hs + a_0 V since this term does not contribute to the bath hierarchy

    Args:
        poly_coeffs (List[complex]): bath polynomial coefficients in reverse order (numpy convention)
        Nmax_Omega (int): maximum order of Omega_n to compute
        Nmax_tilde_Omega (int): maximum order of tilde_Omega_n to compute
        Hs (np.ndarray): the system Hamiltonian
        V (np.ndarray): the system-bath interaction mode
        mu (np.ndarray): the interested system operator
        sigma_0 (np.ndarray): the initial system density matrix
        theta_func (Callable[[int], complex]): theta function
        eta_func (Callable[[int], complex]): eta function
        expval_func (Callable[[str], complex]): expectation value wrapper function
        njobs (int, optional): number of jobs for parallel computation. Defaults to -1.
        innermax (int, optional): maximum number of threads for each job. Defaults to 1.
    """
    # Define new Hs' = Hs + a_0 V
    Hsp = Hs + poly_coeffs[-1] * V

    # The zeroth order term, the mu operator
    zeroth_terms = [SBGeneralTerm(op=mu, bathpoly=BathPolynomial())]

    # The iLv_nth_mu terms
    iLv_nth_mu = deepcopy(zeroth_terms)

    # The QiLv_nth_mu terms
    QiLv_nth_mu = deepcopy(zeroth_terms)

    # The iLv_QiLv_nth_mu terms
    iLv_QiLv_nth_mu, tilde_Omega_n = apply_iLv(
        QiLv_nth_mu,
        Hsp, V, mu, sigma_0,
        theta_func, eta_func, expval_func,
        poly_coeffs,
        njobs, innermax,
        quantum = quantum)  # iLv (QiLv)^n mu, n = 0 at this point

    # On-the-fly writers
    # K1Writer = MomentsWriter("K1_n.dat", flag='K1n')
    if quantum:
        tildeOmegaWriter = MomentsWriter(
            "tilde_moments_quantum.dat", flag='tilde_moments_quantum')
        OmegaWriter = MomentsWriter(
            "moments_quantum.dat", flag='moments_quantum')
    else:
        tildeOmegaWriter = MomentsWriter(
            "tilde_moments_classical.dat", flag='tilde_moments_classical')
        OmegaWriter = MomentsWriter(
            "moments_classical.dat", flag='moments_classical')

    print("Start Computing moments Omega_n and tilde_Omega_n")
    print()

    Nmax = max(Nmax_Omega, Nmax_tilde_Omega)
    start_prog = time.perf_counter()
    print(tab_str("Timings"))
    for order in range(1, Nmax+1):
        if order <= Nmax_Omega:
            start = time.perf_counter()
            iLv_nth_mu, Omega_n = apply_iLv(
                iLv_nth_mu,
                Hsp, V, mu, sigma_0,
                theta_func, eta_func, expval_func,
                poly_coeffs,
                njobs, innermax,
                quantum = quantum)
            time_comm_and_project_Omega = time.perf_counter() - start
            OmegaWriter.write_line(Omega_n)

            # logging
            sheader = f"[Omega_{order:02d}]: "
            # stime = f"Terms = {len(iLv_nth_mu):8d}. Time = {(time_comm_Omega + time_project_Omega):8.1f}"
            stime = f"Terms = {len(iLv_nth_mu):8d}. Time = {(time_comm_and_project_Omega):8.1f}"
            s = f"{sheader:>20s}{stime:<}"
            print(tab_str(s, char=' '))

        # print()
        # print("Order = ", order, ". Terms: ")
        # for term in iLv_nth_mu:
        #     print(term)
        # print()

        if order <= Nmax_tilde_Omega:
            start = time.perf_counter()
            iLv_QiLv_nth_mu, tilde_Omega_n = apply_QiLv(
                iLv_QiLv_nth_mu, tilde_Omega_n,
                Hsp, V, mu, sigma_0,
                theta_func, eta_func, expval_func,
                poly_coeffs,
                njobs, innermax, quantum = quantum)
            time_comm_and_proj_K1 = time.perf_counter() - start
            tildeOmegaWriter.write_line(tilde_Omega_n)
            sheader = f"[tilde_Omega_{order:02d}]: "
            stime = f"Terms = {len(iLv_QiLv_nth_mu):8d}. Time = {time_comm_and_proj_K1:8.1f}"
            s = f"{sheader:>20s}{stime:<}"
            print(tab_str(s, char=' '))

    bottom = "!" + "=" * 71 + "!"
    print(bottom)

    # Close the writers
    OmegaWriter.close()
    tildeOmegaWriter.close()

    # Total time for the program
    total_time = time.perf_counter() - start_prog
    print(f"Total time elapsed: {total_time:.2f} seconds")
