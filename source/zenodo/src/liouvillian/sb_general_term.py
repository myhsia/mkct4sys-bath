import numpy as np
from liouvillian.bath_mode import BathMode
from liouvillian.bath_polynomial import BathPolynomial
from liouvillian.utils import comm

from dataclasses import dataclass
from typing import List, Callable
from copy import deepcopy


@dataclass
class SBGeneralTerm:
    # op: NDArray[np.complex128]
    op: np.ndarray
    bathpoly: BathPolynomial

    def __str__(self):
        # get the polynomial string
        poly_str = self.bathpoly._str_poly().strip()
        op_str = f"{self.op}"
        out_str = "===\n"
        out_str += f"{poly_str}:\n"
        out_str += f"{op_str}\n"
        out_str += "===\n"
        return out_str

    def project_to_mu(
        self,
        mu: np.ndarray,
        sigma_0: np.ndarray,
        expval_func: Callable[[str], complex]
    ) -> np.ndarray:
        denominator = np.trace(mu @ mu @ sigma_0)
        expval_sys = np.trace(self.op @ mu @ sigma_0) / denominator
        expval_bath = self.bathpoly.expval(expval_func)
        return expval_sys * expval_bath

    def apply_iLS(
        self,
        Hs: np.ndarray
    ) -> List['SBGeneralTerm']:
        new_op = 1.j * comm(Hs, self.op)
        return [SBGeneralTerm(op=new_op, bathpoly=deepcopy(self.bathpoly))]

    def apply_iLB(
        self,
        theta_func: Callable[[int], complex]
    ) -> List['SBGeneralTerm']:
        new_bathpoly_list = self.bathpoly.apply_iLB(theta_func)
        new_general_list = []
        for poly in new_bathpoly_list:
            coeff = poly.coeff
            new_op = coeff * self.op
            poly.coeff = 1.0
            new_general_list.append(SBGeneralTerm(op=new_op, bathpoly=poly))
        return new_general_list

    def apply_iLSB(
        self,
        poly_coeffs: List[complex],
        V: np.ndarray,
        theta_func: Callable[[int], complex]
    ) -> List['SBGeneralTerm']:
        # apply the iLSB superoperator to the current term
        # iLSB G
        # = i.j * [V (a_1 x + a_2 x^2 + ... + a_n x^n), op bathpoly]
        # = 1.j * [V, op] \otimes (a_1 x + a_2 x^2 + ... + a_n x^n) bathpoly + 1.j * op V \otimes [a_1 x + a_2 x^2 + ... + a_n x^n, bathpoly]
        # = 1.j * [V op - op V + op V] \otimes (a_1 x + a_2 x^2 + ... + a_n x^n) bathpoly - 1.j * op V \otimes bathpoly * (a_1 x + a_2 x^2 + ... + a_n x^n)
        a_n = poly_coeffs[:-1]
        order_list = reversed(range(1, len(a_n) + 1))
        poly_interactions = []
        for a, n in zip(a_n, order_list):
            # if np.allclose(a, 0.0, atol=1e-10):
            #     continue
            modes = [BathMode(1, 0) for _ in range(n)]
            poly = BathPolynomial(coeff=a, pos_modes=modes, mom_modes=[])
            poly_interactions.append(poly)



        new_general_list = []
        # first term
        _new_op1 = 1.j * np.dot(V, self.op)
        new_bathpoly1_list = []
        for poly in poly_interactions:
            new_bathpoly1_list += self.bathpoly.left_multiply_poly(
                poly, theta_func)

        for poly in new_bathpoly1_list:
            new_op1 = _new_op1 * poly.coeff
            poly.coeff = 1.0
            new_general_list.append(SBGeneralTerm(op=new_op1, bathpoly=poly))


        # second term
        # new_bathpoly2_list = self.bathpoly.apply_comm_rho0(theta_func)
        _new_op2 = -1.j * np.dot(self.op, V)
        new_bathpoly2_list = []
        for poly in poly_interactions:
            new_bathpoly2_list += self.bathpoly.right_multiply_poly(
                poly, theta_func)

        for poly in new_bathpoly2_list:
            new_op2 = _new_op2 * poly.coeff
            poly.coeff = 1.0
            new_general_list.append(SBGeneralTerm(op=new_op2, bathpoly=poly))

        return self.combine(new_general_list)


    @staticmethod
    def combine(term_list: List['SBGeneralTerm']) -> List['SBGeneralTerm']:
        # get the unique polynomials
        poly_str_list = [str(term.bathpoly) for term in term_list]
        unique_str_list = list(set(poly_str_list))

        # use a dictionary to store the new terms
        new_term_dict = {s: None for s in unique_str_list}

        for term in term_list:
            s = str(term.bathpoly)
            if new_term_dict[s] is None:
                new_term_dict[s] = deepcopy(term)
                new_term_dict[s].op = term.op * term.bathpoly.coeff
                new_term_dict[s].bathpoly.coeff = 1.0
            else:
                new_term_dict[s].op += term.op * term.bathpoly.coeff
        return list(new_term_dict.values())

    @staticmethod
    def combine_pure_system(
        term_list: List['SBGeneralTerm'],
        pure_system_term: 'SBGeneralTerm'
    ) -> List['SBGeneralTerm']:
        # Assert that the pure_system_term is a pure system term
        str_pure_system = pure_system_term.bathpoly._str_poly()
        assert str_pure_system == "", f"Invalid pure system term: {str_pure_system}"

        # Traverse the term_list and combine the terms with the pure_system_term
        # change inplace
        for ii, term in enumerate(term_list):
            str_term = term.bathpoly._str_poly()
            if str_term == "":
                # Found the pure system term
                term_list[ii].op += pure_system_term.op

                # break the for loop
                break
        return term_list

    def apply_iLv(
        self,
        poly_coeffs: List[complex],
        Hs: np.ndarray,
        V: np.ndarray,
        mu: np.ndarray,
        theta_func: Callable[[int], complex]
    ) -> List['SBGeneralTerm']:
        terms = []
        terms += self.apply_iLS(Hs)
        terms += self.apply_iLB(theta_func)
        terms += self.apply_iLSB(poly_coeffs, V, theta_func)
        return terms
