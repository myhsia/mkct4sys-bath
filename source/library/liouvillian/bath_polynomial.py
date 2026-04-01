import sympy as sp
from liouvillian.bath_mode import BathMode, BathModeCommute, apply_iLB
from typing import List, Tuple, Callable
from copy import deepcopy
from dataclasses import dataclass, field


@dataclass
class BathPolynomial:
    """Single term of a general bath polynomial
    BathPolynomial({BathMode}) =
        \prod_i BathMode_i(n_i, 1) \prod_j BathMode_j(n_j, -1)
    Note that the convention is to put the position modes on the left,
    and the momentum modes on the right.
    """
    coeff: complex = 1.0
    pos_modes: List[BathMode] = field(default_factory=list)
    mom_modes: List[BathMode] = field(default_factory=list)

    def sort(self, ):
        self.sort_pos()
        self.sort_mom()

    def sort_pos(self, ):
        # sort the current position modes
        self.pos_modes.sort(key=lambda x: x.n)

    def sort_mom(self, ):
        # sort the current momentum modes
        self.mom_modes.sort(key=lambda x: x.n)

    def _str_coeff(self):
        # the coeff part
        coeff_str = f"{self.coeff:6.3E} "
        return coeff_str

    def _str_poly(self):
        # the polynomail part
        poly_str = ""
        for mode in self.pos_modes:
            poly_str += str(mode) + ' '
        for mode in self.mom_modes:
            poly_str += str(mode) + ' '
        return poly_str

    def __str__(self):
        return self._str_coeff() + self._str_poly()

    @staticmethod
    def combine(poly_list: List['BathPolynomial']) -> List['BathPolynomial']:
        # get the unique terms
        poly_str_list = [str(poly) for poly in poly_list]
        unique_str_list = list(set(poly_str_list))

        # use a dictionary to store the new polynomial
        new_poly_dict = {s: None for s in unique_str_list}

        for poly in poly_list:
            s = str(poly)
            if new_poly_dict[s] is None:
                new_poly_dict[s] = deepcopy(poly)
            else:
                new_poly_dict[s].coeff += poly.coeff

        # dictionary to list
        new_poly_list = list(new_poly_dict.values())
        return new_poly_list

    @staticmethod
    def move_momentum_right(
        pi: BathMode,
        pos_modes: List[BathMode],
        theta_func: Callable[[int], complex],
        quantum: bool = True
    ) -> Tuple[List[complex], List[List[BathMode]]]:
        # first compute the commutator of pi with all the position operators
        coeff_list = []
        for rho in pos_modes:
            comm = BathModeCommute(pi, rho, theta_func, quantum = quantum)
            coeff_list.append(comm)

        # construct the new mode list
        mode_list = []
        for i in range(len(coeff_list)):
            new_pos_modes = pos_modes[:i] + pos_modes[i+1:]
            mode_list.append(new_pos_modes)
        return coeff_list, mode_list

    @staticmethod
    def move_position_left(
        rho: BathMode,
        mom_modes: List[BathMode],
        theta_func: Callable[[int], complex],
        quantum: bool = True
    ) -> Tuple[List[complex], List[List[BathMode]]]:
        # first compute the commutator of rho with all the momentum operators
        comm_list = []
        for pi in mom_modes:
            comm = BathModeCommute(pi, rho, theta_func, quantum = quantum)
            comm_list.append(comm)

        # construct the new mode list
        mode_list = []
        for i in range(len(comm_list)):
            new_mom_modes = mom_modes[:i] + mom_modes[i+1:]
            mode_list.append(new_mom_modes)

        return comm_list, mode_list

    def left_multiply_poly(
        self,
        other: 'BathPolynomial',
        theta_func: Callable[[int], complex],
        quantum: bool = True
    ) -> List['BathPolynomial']:
        other_pos_modes = other.pos_modes
        other_mom_modes = other.mom_modes
        other_coeff = other.coeff

        # new_poly = deepcopy(self)
        tmp = deepcopy(self)

        # For the momentum operators in the other polynomial
        # recursively apply the left_multiply_mode function
        new_poly_list = [tmp]
        for mode in reversed(other_mom_modes):
            new_poly_list_new = []
            for poly in new_poly_list:
                new_poly_list_new += poly.left_multiply_mode(
                    mode, theta_func, quantum = quantum)
            new_poly_list = new_poly_list_new

        # For each position operator in the other polynomial
        # just append the position operator to the current polynomial
        for poly in new_poly_list:
            poly.pos_modes += other_pos_modes
            poly.sort_pos()
            poly.coeff *= other_coeff

        return self.combine(new_poly_list)

    def left_multiply_mode(
        self,
        mode: BathMode,
        theta_func: Callable[[int], complex],
        quantum: bool = True
    ) -> List['BathPolynomial']:
        if (mode.sig == 1):
            # if the signature is 1 (position), then simply
            # append the mode to the list of position modes
            # and the sort the list
            new_poly = deepcopy(self)
            new_poly.pos_modes.append(mode)
            new_poly.sort_pos()
            return [new_poly]

        else:
            if len(self.pos_modes) == 0:
                # no position operator in the left
                new_poly = deepcopy(self)
                new_poly.mom_modes.append(mode)
                new_poly.sort_mom()
                return [new_poly]
            else:
                # get the coeffs and the new mode list
                coeff_list, modes_list = self.move_momentum_right(
                    mode, self.pos_modes, theta_func, quantum = quantum)
                n_terms = len(coeff_list)

                new_poly_list = []
                for i in range(n_terms):
                    new_poly = BathPolynomial(
                        coeff=self.coeff * coeff_list[i],
                        pos_modes=modes_list[i],
                        mom_modes=deepcopy(self.mom_modes)
                    )
                    new_poly_list.append(new_poly)

                new_poly = deepcopy(self)
                new_poly.mom_modes.append(mode)
                new_poly.sort_mom()
                new_poly_list.append(new_poly)

                return self.combine(new_poly_list)

    def right_multiply_poly(
        self,
        other: 'BathPolynomial',
        theta_func: Callable[[int], complex],
        quantum: bool = True
    ) -> List['BathPolynomial']:
        # For each momentum operator in the other polynomial
        # just append the momentum operator to the current polynomial
        tmp = deepcopy(self)

        # new_poly = deepcopy(self)
        # new_poly.mom_modes += other.mom_modes
        # new_poly.sort_mom()

        # For the position operators in the other polynomial
        # recursively apply the right_multiply_mode function
        new_poly_list = [tmp]
        for mode in other.pos_modes:
            new_poly_list_new = []
            for poly in new_poly_list:
                new_poly_list_new += poly.right_multiply_mode(
                    mode, theta_func, quantum = quantum)
            new_poly_list = new_poly_list_new

        # For each momentum operator in the other polynomial
        # just append the momentum operator to the current polynomial
        for poly in new_poly_list:
            poly.mom_modes += other.mom_modes
            poly.sort_mom()
            poly.coeff *= other.coeff

        return self.combine(new_poly_list)

    def right_multiply_mode(
        self,
        mode: BathMode,
        theta_func: Callable[[int], complex],
        quantum: bool = True
    ) -> List['BathPolynomial']:
        if mode.sig == -1:
            # if the signature is -1 (momentum), then simply
            # append the mode to the list of momentum modes
            # and the sort the list
            new_poly = deepcopy(self)
            new_poly.mom_modes.append(mode)
            new_poly.sort_mom()
            return [new_poly]
        else:
            # if there is no momentum operator on the right
            # then we simply append the position operator to the list
            if len(self.mom_modes) == 0:
                new_poly = deepcopy(self)
                new_poly.pos_modes.append(mode)
                new_poly.sort_pos()
                return [new_poly]
            else:
                # get the coeffs and the new mode list
                coeff_list, modes_list = self.move_position_left(
                    mode, self.mom_modes, theta_func, quantum = quantum)
                n_terms = len(coeff_list)

                new_poly_list = []
                for i in range(n_terms):
                    new_poly = BathPolynomial(
                        coeff=self.coeff * coeff_list[i],
                        pos_modes=deepcopy(self.pos_modes),
                        mom_modes=modes_list[i]
                    )
                    new_poly_list.append(new_poly)

                new_poly = deepcopy(self)
                new_poly.pos_modes.append(mode)
                new_poly.sort_pos()
                new_poly_list.append(new_poly)

                return self.combine(new_poly_list)

    def apply_iLB(
        self,
        theta_func: Callable[[int], complex],
        quantum: bool = True,
    ) -> List['BathPolynomial']:
        # apply the iLB superoperator to the current polynomial
        # iLB = 1/2 \sum_j [p_j^2, BathPolynomial] + [w_j^2 q_j^2, BathPolynomial]
        # 1/2 \sum_j [p_j^2, BathPolynomial] = 1/2 \sum_j [p_j^2, BathPolynomial.Position] * BathPolynomial.Momentum
        # 1/2 \sum_j [w_j^2 q_j^2, BathPolynomial] = BathPolynomial.Position * 1/2 \sum_j [w_j^2 q_j^2, BathPolynomial.Momentum]

        pos_modes = self.pos_modes
        mom_modes = self.mom_modes
        n_pos, n_mom = len(pos_modes), len(mom_modes)
        all_modes = pos_modes + mom_modes
        types = [0] * len(pos_modes) + [1] * len(mom_modes)
        new_poly_list = []
        for ii, (tp, mode) in enumerate(zip(types, all_modes)):
            modes_before = all_modes[:ii]
            modes_after = all_modes[ii+1:]
            coeff, new_mode = apply_iLB(mode)
            if tp == 0:
                # The new mode is a momentum operator
                # For normal ordering, we need to move the new momentum operator to the left
                new_mom_mode = new_mode
                pos_modes_before = modes_before
                n_pos_after = n_pos - len(pos_modes_before) - 1
                pos_modes_after = modes_after[:n_pos_after]

                # move the new momentum mode to the right of pos_modes_after
                coeff_list, mode_list = self.move_momentum_right(
                    new_mom_mode,
                    pos_modes_after,
                    theta_func,
                    quantum = quantum)
                for i in range(len(coeff_list)):
                    coeff_i = self.coeff * coeff * coeff_list[i]
                    pos_modes_i = pos_modes_before + mode_list[i]
                    new_poly = BathPolynomial(
                        coeff=coeff_i,
                        pos_modes=pos_modes_i,
                        mom_modes=deepcopy(mom_modes)
                    )
                    new_poly_list.append(new_poly)

                coeff_last = self.coeff * coeff
                pos_modes_last = pos_modes_before + pos_modes_after
                mom_modes_last = deepcopy(mom_modes)
                mom_modes_last.append(new_mom_mode)
                new_poly = BathPolynomial(
                    coeff=coeff_last,
                    pos_modes=pos_modes_last,
                    mom_modes=mom_modes_last
                )
                new_poly_list.append(new_poly)

            elif tp == 1:
                # The new mode is a position operator
                # For normal ordering, we need to move the new position operator to the right
                new_pos_mode = new_mode
                mom_modes_after = modes_after
                mom_modes_before = modes_before[n_pos:]

                # move the new position mode to the left of mom_modes_before
                coeff_list, mode_list = self.move_position_left(
                    new_pos_mode,
                    mom_modes_before,
                    theta_func,
                    quantum = quantum)
                for i in range(len(coeff_list)):
                    coeff_i = self.coeff * coeff * coeff_list[i]
                    mom_modes_i = mode_list[i] + mom_modes_after
                    new_poly = BathPolynomial(
                        coeff=coeff_i,
                        pos_modes=deepcopy(pos_modes),
                        mom_modes=mom_modes_i
                    )
                    new_poly_list.append(new_poly)

                coeff_last = self.coeff * coeff
                mom_modes_last = mom_modes_before + mom_modes_after
                pos_modes_last = deepcopy(pos_modes)
                pos_modes_last.append(new_pos_mode)
                new_poly = BathPolynomial(
                    coeff=coeff_last,
                    pos_modes=pos_modes_last,
                    mom_modes=mom_modes_last
                )
                new_poly_list.append(new_poly)

        return self.combine(new_poly_list)

    def apply_comm_rho0(
        self,
        theta_func: Callable[[int], complex],
        quantum: bool = True
    ) -> List['BathPolynomial']:
        # apply the commutator with the rho_0 operator
        # [rho_0, BathPolynomial]
        pos_modes = self.pos_modes
        mom_modes = self.mom_modes

        # if there is no momenta operators, then the commutator is zero
        if len(mom_modes) == 0:
            return []

        # if there is momenta operators, then the commutator is
        # [rho_0, p_n1 p_n2 ... ] = [rho_0, p_n1] p_n2 ... + p_n1 [rho_0, p_n2] ...
        new_poly_list = []
        rho0 = BathMode(sig=1, n=0)
        for ii, pi in enumerate(mom_modes):
            # compute the commutator
            comm = BathModeCommute(rho0, pi, theta_func, quantum = quantum)
            # comm = BathModeCommute(pi, rho0)
            new_mom_list = mom_modes[:ii] + mom_modes[ii+1:]
            new_poly = BathPolynomial(
                coeff=self.coeff*comm,
                pos_modes=deepcopy(self.pos_modes),
                mom_modes=new_mom_list,
            )
            new_poly_list.append(new_poly)
        return self.combine(new_poly_list)

    def expval(
        self,
        expval_func: Callable[[str], complex],
    ) -> complex:
        # evaluate the thermal expectation value of the current polynomial

        # first compute the order of the polynomial
        n = len(self.pos_modes) + len(self.mom_modes)

        if n % 2 != 0:
            return 0.0
        else:
            # modes = self.pos_modes + self.mom_modes
            modes = self._str_poly()

            assert self.coeff == 1.0, "The coefficient should be 1.0 at the expval stage"
            return expval_func(modes)
