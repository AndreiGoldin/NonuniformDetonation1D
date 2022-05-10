# Contains classes with different numerical methods to approximate spatial fluxes
from abc import ABC, abstractmethod
import numpy as np
import numba as nb


class SpaceMethod:
    possible_methods = {}

    @classmethod
    def register_method(cls, method_type):
        def decorator(subclass):
            cls.possible_methods[method_type] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, method_type, params):
        if method_type not in cls.possible_methods:
            raise ValueError(f'Unknown method: {method_type}. Possible options are {list(cls.possible_methods.keys())}.')
        return cls.possible_methods[method_type](params)

    @abstractmethod
    def flux_approx():
        pass

    def __repr__(self):
        return f'This is the {self.__class__.__name__} method with parameters:\n {self.parameters}'


@SpaceMethod.register_method('Upwind')
class Upwind(SpaceMethod):
    def __init__(self, params):
        self.parameters = params

    def flux_approx(self, array, speed):
        flux_approx = np.copy(array)
        flux_approx[:, 1:-1] = -speed[:, 1:-1]*(array[:, 2:]-array[:, 1:-1])*(speed[:, 1:-1]<0.) \
                - speed[:, 1:-1]*(array[:, 1:-1]-array[:, :-2])*(speed[:, 1:-1]>=0.)
        return flux_approx


@SpaceMethod.register_method('UpstreamCentral')
class UpstreamCentral(SpaceMethod):
    def __init__(self, params):
        self.parameters = params

    def flux_approx(self, nodeim2,nodeim1,nodei,nodeip1,nodeip2):
        flux_approx = 1/60.*(2.*nodeim2-13.*nodeim1+47.*nodei+27.*nodeip1-3.*nodeip2)
        return flux_approx


@SpaceMethod.register_method('WENO5M')
class WENO5M(SpaceMethod):
    def __init__(self, params):
        if 'eps' not in params.keys():
            raise AttributeError(f'Parameter eps is not defined for WENO5M method')
        self.parameters = params
        self.eps = params['eps']
        self.p = params['p']

    def source_approx():
        pass

    @staticmethod
    @nb.vectorize("f8(f8,f8)", nopython=True)
    def g_numba(w_k, omega):
        """
        Calculate the weighted coefficients according to Henrick et al. (2006)
        """
        return ( omega * (w_k + w_k**2 - 3. * w_k * omega + omega**2) / (w_k**2 + (1. - 2. * w_k) * omega))

    def _flux_approx(self, nodeim2, nodeim1, nodei, nodeip1, nodeip2):
        # Interpolation of the fluxes on substencils
        q_im1 = 1.0 / 3.0 * nodeim2 - 7.0 / 6.0 * nodeim1 + 11.0 / 6.0 * nodei
        q_i = -1.0 / 6.0 * nodeim1 + 5.0 / 6.0 * nodei + 1.0 / 3.0 * nodeip1
        q_ip1 = 1.0 / 3.0 * nodei + 5.0 / 6.0 * nodeip1 - 1.0 / 6.0 * nodeip2
        # Indicators of smoothness
        beta_im1 = ( 13.0 / 12.0 * (nodeim2 - 2.0 * nodeim1 + nodei) ** 2
            + 0.25 * (nodeim2 - 4.0 * nodeim1 + 3.0 * nodei) ** 2)
        beta_i = ( 13.0 / 12.0 * (nodeim1 - 2.0 * nodei + nodeip1) ** 2
            + 0.25 * (nodeip1 - nodeim1) ** 2)
        beta_ip1 = ( 13.0 / 12.0 * (nodei - 2.0 * nodeip1 + nodeip2) ** 2
            + 0.25 * (3.0 * nodei - 4.0 * nodeip1 + nodeip2) ** 2)
        # Weights for stencils
        alpha_im1 = 1.0 / 10.0 / (self.eps + beta_im1) ** self.p
        alpha_i = 6.0 / 10.0 / (self.eps + beta_i) ** self.p
        alpha_ip1 = 3.0 / 10.0 / (self.eps + beta_ip1) ** self.p
        alpha_sum = alpha_im1 + alpha_i + alpha_ip1
        # Modified weights
        mod_omega_0 = self.g_numba(1.0 / 10.0, alpha_im1 / alpha_sum)
        mod_omega_1 = self.g_numba(6.0 / 10.0, alpha_i / alpha_sum)
        mod_omega_2 = self.g_numba(3.0 / 10.0, alpha_ip1 / alpha_sum)
        mod_omega_sum = mod_omega_0 + mod_omega_1 + mod_omega_2
        # Numerical flux
        modified_weno5_flux = ( mod_omega_0 * q_im1 + mod_omega_1 * q_i + mod_omega_2 * q_ip1) / mod_omega_sum
        return modified_weno5_flux


    def flux_approx(self):
        eps = self.eps
        g_numba = self.g_numba
        def inner(nodeim2, nodeim1, nodei, nodeip1, nodeip2):
            # Interpolation of the fluxes on substencils
            q_im1 = 1.0 / 3.0 * nodeim2 - 7.0 / 6.0 * nodeim1 + 11.0 / 6.0 * nodei
            q_i = -1.0 / 6.0 * nodeim1 + 5.0 / 6.0 * nodei + 1.0 / 3.0 * nodeip1
            q_ip1 = 1.0 / 3.0 * nodei + 5.0 / 6.0 * nodeip1 - 1.0 / 6.0 * nodeip2
            # Indicators of smoothness
            beta_im1 = ( 13.0 / 12.0 * (nodeim2 - 2.0 * nodeim1 + nodei) ** 2
                + 0.25 * (nodeim2 - 4.0 * nodeim1 + 3.0 * nodei) ** 2)
            beta_i = ( 13.0 / 12.0 * (nodeim1 - 2.0 * nodei + nodeip1) ** 2
                + 0.25 * (nodeip1 - nodeim1) ** 2)
            beta_ip1 = ( 13.0 / 12.0 * (nodei - 2.0 * nodeip1 + nodeip2) ** 2
                + 0.25 * (3.0 * nodei - 4.0 * nodeip1 + nodeip2) ** 2)
            # Weights for stencils
            alpha_im1 = 1.0 / 10.0 / (eps + beta_im1) ** 2
            alpha_i = 6.0 / 10.0 / (eps + beta_i) ** 2
            alpha_ip1 = 3.0 / 10.0 / (eps + beta_ip1) ** 2
            alpha_sum = alpha_im1 + alpha_i + alpha_ip1
            # Modified weights
            mod_omega_0 = g_numba(1.0 / 10.0, alpha_im1 / alpha_sum)
            mod_omega_1 = g_numba(6.0 / 10.0, alpha_i / alpha_sum)
            mod_omega_2 = g_numba(3.0 / 10.0, alpha_ip1 / alpha_sum)
            mod_omega_sum = mod_omega_0 + mod_omega_1 + mod_omega_2
            # Numerical flux
            modified_weno5_flux = (mod_omega_0 * q_im1 + mod_omega_1 * q_i + mod_omega_2 * q_ip1) / mod_omega_sum
            return modified_weno5_flux
        return inner


class TENO(SpaceMethod):
    def __init__(self, params):
        pass

    def flux_approx(self):
        pass


if __name__=='__main__':
    print('Testing SpaceMethod factory')
    try:
        method = SpaceMethod.create('WENO5M',{})
    except Exception as e:
        print(f'Exception occured: {e}')
    try:
        method = SpaceMethod.create('WENO5M',{'eps':1e-40})
    except Exception as e:
        print(f'Exception occured: {e}')
    else:
        print(f'Success! {method}')
    state = np.arange(0,101)
    fluxes = method.flux_approx(state,state,state,state,state)
    print(fluxes)
