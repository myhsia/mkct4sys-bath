import numpy as np
from joblib import Parallel, delayed, parallel_config
from liouvillian.sb_general_term import SBGeneralTerm
from liouvillian.bath_polynomial import BathPolynomial
from liouvillian.utils import comm
from typing import List, Callable


def apply_iLv(
    terms: List[SBGeneralTerm],
    Hs: np.ndarray,
    V: np.ndarray,
    mu: np.ndarray,
    sigma_0: np.ndarray,
    theta_func: Callable[[int], complex],
    eta_func: Callable[[int], complex],
    expval_func: Callable[[str], complex],
    poly_coeffs: List[complex] = None,
    njobs: int = -1,
    innermax: int = 1,
) -> List[SBGeneralTerm]:

    if poly_coeffs is None:
        poly_coeffs = [1.0, 0.0]  # meaning H_SB = V q

    def single_term_iLv(term: SBGeneralTerm) -> List[SBGeneralTerm]:
        return term.apply_iLv(poly_coeffs, Hs, V, mu, theta_func)

    def single_term_proj(term: SBGeneralTerm) -> complex:
        return term.project_to_mu(mu, sigma_0, expval_func)

    with parallel_config(backend="loky", inner_max_num_threads=innermax):
        tmp = Parallel(n_jobs=njobs, return_as='generator_unordered')(
            delayed(single_term_iLv)(term) for term in terms)

    new_terms = [item for sublist in tmp for item in sublist]
    new_terms = SBGeneralTerm.combine(new_terms)

    for term in new_terms:
        term.bathpoly.sort()

    with parallel_config(backend="loky", inner_max_num_threads=innermax):
        tmp = Parallel(n_jobs=njobs, return_as='generator_unordered')(
            delayed(single_term_proj)(term) for term in new_terms)
    Omega_n = sum(tmp)

    return new_terms, Omega_n


def apply_QiLv(
    iLv_QiLv_nth_mu: List[SBGeneralTerm],
    tilde_Omega: complex,
    Hs: np.ndarray,
    V: np.ndarray,
    mu: np.ndarray,
    sigma_0: np.ndarray,
    theta_func: Callable[[int], complex],
    eta_func: Callable[[int], complex],
    expval_func: Callable[[str], complex],
    poly_coeffs: List[complex] = None,
    njobs: int = -1,
    innermax: int = 1,
) -> List[SBGeneralTerm]:
    p_term = SBGeneralTerm(op=-mu*tilde_Omega, bathpoly=BathPolynomial())
    QiLv_QiLv_n_mu = SBGeneralTerm.combine_pure_system(iLv_QiLv_nth_mu, p_term)
    return apply_iLv(QiLv_QiLv_n_mu,
                     Hs, V, mu, sigma_0,
                     theta_func, eta_func, expval_func,
                     poly_coeffs=poly_coeffs,
                     njobs=njobs, innermax=innermax)
