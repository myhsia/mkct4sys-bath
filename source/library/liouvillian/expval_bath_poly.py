import sympy as sp

from liouvillian.utils import parse_reps

from dataclasses import dataclass
from typing import Callable

type_i2c = Callable[[int], complex]


def expval_BathPoly(
    bp_str: str,
    theta: type_i2c,
    eta: type_i2c,
    quantum: bool = True
) -> complex:
    # get the string representation of the bath polynomial
    modes = bp_str.split()
    m_list = []
    n_list = []
    for mode in modes:
        sig, n = mode.split('_')
        if sig == 'ρ':
            m_list.append(int(n))
        else:
            n_list.append(int(n))

    m_list.sort()
    n_list.sort()

    # concatenate the repeated modes
    m_list, m_reps = parse_reps(m_list)
    n_list, n_reps = parse_reps(n_list)

    # get the number of position and momentum operators
    n = sum(m_reps) + sum(n_reps)

    # if the number of modes is odd, then the expectation value is zero
    if n % 2 != 0:
        return 0.0
    elif n == 0:
        return 1.0
    else:
        # generate the symbols for the generating function
        l_list = [sp.symbols(f'l_{i}') for i in range(len(m_list))]   # lambda list
        lp_list = [sp.symbols(f'lp_{i}') for i in range(len(n_list))] # lambda prime list

        # compute the expression for the generating function
        expo = 0
        eta_dict = {}
        theta_dict = {}
        for i, (li, mi) in enumerate(zip(l_list, m_list)):
            for j, (lj, mj) in enumerate(zip(l_list, m_list)):
                order = mi + mj
                if order not in eta_dict.keys():
                    eta_ij = sp.symbols(f'eta{order}', real=True)
                    eta_dict[order] = eta_ij
                else:
                    eta_ij = eta_dict[order]
                expo += li * lj * eta_ij / 2

        for i, (lpi, ni) in enumerate(zip(lp_list, n_list)):
            for j, (lpj, nj) in enumerate(zip(lp_list, n_list)):
                order = ni + nj
                if order not in eta_dict.keys():
                    eta_ij = sp.symbols(f'eta{order}', real=True)
                    eta_dict[order] = eta_ij
                else:
                    eta_ij = eta_dict[order]
                expo += lpi * lpj * eta_ij / 2
        if quantum:
            for i, (li, mi) in enumerate(zip(l_list, m_list)):
                for j, (lpj, nj) in enumerate(zip(lp_list, n_list)):
                    order = mi + nj
                    if order not in theta_dict.keys():
                        theta_ij = sp.symbols(f'theta{order}', real=True)
                        theta_dict[order] = theta_ij
                    else:
                        theta_ij = theta_dict[order]
                    expo += li * lpj * theta_ij * sp.I / 2
        expr = sp.exp(expo)

        # compute the derivative sequence
        all_sbs = l_list + lp_list
        # for sb in reversed(all_sbs):
        # for sb, reps in zip(all_sbs, m_reps + n_reps):
        for sb, reps in reversed(list(zip(all_sbs, m_reps + n_reps))):
            # expr = expr.diff(sb)
            for _ in range(reps):
                expr = expr.diff(sb)
            # evaluate the sb after all the derivatives are taken
            expr = expr.subs(sb, 0)

        # for sb in all_sbs:
        #     expr = expr.subs(sb, 0)

        # Evaluate all eta and theta symbols
        sub_dict = {}
        for key, val in eta_dict.items():
            sub_dict[val] = eta(key)
        for key, val in theta_dict.items():
            sub_dict[val] = theta(key)

        # Evaluate the eta and theta symbols
        val = expr.subs(sub_dict)
        return complex(val)
