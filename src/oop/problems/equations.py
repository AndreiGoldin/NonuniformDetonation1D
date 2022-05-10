# Contains classes with different hyperbolic equations and systems
from abc import ABC, abstractmethod
import numpy as np
from scipy import integrate


class Equations:
    possible_equations = {}

    @classmethod
    def register_equations(cls, equations_type):
        def decorator(subclass):
            cls.possible_equations[equations_type] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, equations_type, params):
        if equations_type not in cls.possible_equations:
            raise ValueError(f'Unknown equations: {equations_type}. Possible '+
                    'options are {list(cls.possible_equations.keys())}.')
        return cls.possible_equations[equations_type](params)

    @abstractmethod
    def convert_to_phys_vars(self, array):
        pass

    @abstractmethod
    def convert_to_cons_vars(self, array):
        pass

    @abstractmethod
    def calculate_fluxes(self, array):
        pass

    @abstractmethod
    def calculate_sources(self, array):
        pass

    def __repr__(self):
        return f'This is the {self.__class__.__name__} equations with parameters:\n {self.parameters}'


@Equations.register_equations('Advection')
class Advection(Equations):
    def __init__(self, params={'speed':1.0}):
        self.parameters = params
        self.speed = params['speed']

    def convert_to_phys_vars(self, array):
        return array

    def convert_to_cons_vars(self, array):
        return array

    def calculate_fluxes(self, array):
        return self.speed*array

    def calculate_sources(self, array):
        return np.zeros_like(array)


@Equations.register_equations('Burgers')
class Burgers(Equations):
    def __init__(self, params):
        self.parameters = params

    def convert_to_phys_vars(self, array):
        return array

    def convert_to_cons_vars(self, array):
        return array

    def calculate_fluxes(self, array):
        return array*array/2.

    def calculate_sources(self, array):
        return np.zeros_like(array)


@Equations.register_equations('Euler')
class Euler(Equations):
    """
    Physical variables: rho, u, p
    Conservative variables: rho, rho*u, rho*(e + u^2/2)
    """
    def __init__(self, params: dict):
        # Prevent the absense of equations' parameters
        for param in ['gamma']:
            if param not in params.keys():
                raise AttributeError(f'Parameter {param} is not defined for the reactive Euler equations.')
        self.gamma = params['gamma']
        self.parameters = params

    def convert_to_phys_vars(self, array):
        """ Transform array of conserved variables to the array of physical
        variables"""
        phys_array = np.copy(array)
        # Velocity
        phys_array[1, :] = array[1, :] / array[0, :]
        # Pressure
        phys_array[2, :] = (self.gamma - 1) * (
            array[2, :]
            - 0.5 * array[1, :] * array[1, :] / array[0, :]
        )
        return phys_array

    def convert_to_cons_vars(self, array):
        """ Transform array of physical variables to the array of conservative
        variables"""
        cons_array = np.copy(array)
        # rho*u
        cons_array[1, :] = array[0, :] * array[1, :]
        # Energy
        cons_array[2, :] = (
            array[2, :] / (self.gamma - 1)
            + 0.5
            * cons_array[1, :]
            * cons_array[1, :]
            / array[0, :]
        )
        return cons_array

    def calculate_fluxes(self, array):
        """ Calculate exact fluxes from the conserved variables """
        flux_array = np.empty_like(array)
        pressure = (self.gamma - 1) * ( array[2, :]
            - 0.5 * array[1, :] * array[1, :] / array[0, :])
        flux_array[0, :] = array[1, :]
        flux_array[1, :] = array[1, :] * array[1, :] / array[0, :] + pressure
        flux_array[2, :] = array[1, :] * (array[2, :] + pressure) / array[0, :]
        return flux_array

    def calculate_sources(self, array):
        """ Calculate the right hand side of the equations from the conserved variables"""
        source = np.zeros_like(array)
        return source

    def calculate_jac_norm(self, array):
        """ Necessary for Lax-Friedrichs splitting for flux approximation
        Input array contains the conservative variables"""
        u = array[1, :] / array[0, :]
        P = (self.gamma - 1.0) * (
            array[2, :] - 0.5 * array[1, :] * array[1, :] / array[0, :])
        sound_speed = np.sqrt(self.gamma*P/array[0,:])
        Evals = np.empty_like(array)
        Evals[0, :] = u - sound_speed
        Evals[1, :] = u
        Evals[2, :] = u + sound_speed
        return np.max(np.abs(Evals), axis=0)


@Equations.register_equations('ReactiveEuler')
class ReactiveEuler(Equations):
    """
    Physical variables: rho, u, p, lambda
    Conservative variables: rho, rho*u, rho*(e + u^2/2), rho*lambda
    """
    def __init__(self, params: dict):
        # Prevent the absense of equations' parameters
        for param in ['act_energy', 'gamma', 'heat_release']:
            if param not in params.keys():
                raise AttributeError(f'Parameter {param} is not defined for the reactive Euler equations.')
        self.act_energy = params['act_energy']
        self.gamma = params['gamma']
        self.heat_release = params['heat_release']
        self.D_CJ = np.sqrt(self.gamma + (self.gamma * self.gamma - 1.0) * self.heat_release / 2.0) + \
                    np.sqrt( (self.gamma * self.gamma - 1.0) * self.heat_release / 2.0)
        self.rate_const = self.calculate_rate_const()
        self.parameters = {**params, **{'rate_const':self.rate_const, 'D_CJ':self.D_CJ} }

    def calculate_rate_const(self):
        gamma, act_energy, Q, D_CJ = self.gamma, self.act_energy, self.heat_release, self.D_CJ
        gp1 = gamma + 1.
        DCJ2 = D_CJ * D_CJ
        DCJ2p1 = DCJ2 + 1.0
        DCJ2mg = DCJ2 - gamma
        def for_reac_rate(y):
            V_lam = gamma / DCJ2  * DCJ2p1 / gp1 * ( 1.0 - DCJ2mg / DCJ2p1 / gamma * (np.sqrt(1 - y)))
            u_lam = 1 / gp1 * DCJ2mg / D_CJ * (1 + np.sqrt(1 - y))
            p_lam = DCJ2p1 / gp1 * ( 1.0 + DCJ2mg / DCJ2p1 * (np.sqrt(1 - y)))
            omega = (1 - y) * np.exp(-act_energy / p_lam / V_lam)
            return np.abs(u_lam - D_CJ) / omega

        rate_const, _ = integrate.quad(for_reac_rate, 0.0, 0.5, epsabs=1e-13, epsrel=1e-13)
        return rate_const

    def convert_to_phys_vars(self, array):
        """ Transform array of conserved variables to the array of physical
        variables"""
        phys_array = np.copy(array)
        # Velocity
        phys_array[1, :] = array[1, :] / array[0, :]
        # Pressure
        phys_array[2, :] = (self.gamma - 1) * (
            array[2, :]
            - 0.5 * array[1, :] * array[1, :] / array[0, :]
            + self.heat_release * array[3, :]
        )
        # Lambda
        phys_array[3, :] = array[3, :] / array[0, :]
        return phys_array

    def convert_to_cons_vars(self, array):
        """ Transform array of physical variables to the array of conservative
        variables"""
        cons_array = np.copy(array)
        # rho*u
        cons_array[1, :] = array[0, :] * array[1, :]
        # rho*lambda
        cons_array[3, :] = array[0, :] * array[3, :]
        # Energy
        cons_array[2, :] = (
            array[2, :] / (self.gamma - 1)
            + 0.5
            * cons_array[1, :]
            * cons_array[1, :]
            / array[0, :]
            - self.heat_release * cons_array[3, :]
        )
        return cons_array

    def calculate_fluxes(self, array):
        """ Calculate exact fluxes from the conserved variables """
        flux_array = np.empty_like(array)
        pressure = (self.gamma - 1) * ( array[2, :]
            - 0.5 * array[1, :] * array[1, :] / array[0, :]
            + self.heat_release * array[3, :])
        flux_array[0, :] = array[1, :]
        flux_array[1, :] = array[1, :] * array[1, :] / array[0, :] + pressure
        flux_array[2, :] = array[1, :] * (array[2, :] + pressure) / array[0, :]
        flux_array[3, :] = array[1, :] * array[3, :] / array[0, :]
        return flux_array

    def calculate_sources(self, array):
        """ Calculate the right hand side of the equations from the conserved variables"""
        pressure = (self.gamma - 1) * ( array[2, :]
            - 0.5 * array[1, :] * array[1, :] / array[0, :]
            + self.heat_release * array[3, :])
        source = np.zeros_like(array)
        source[-1, :] = (
            self.rate_const
            * (array[0, :] - array[3, :])
            * np.exp(-self.act_energy * array[0, :] / pressure)
        )
        return source

    def calculate_jac_norm(self, array):
        """ Necessary for Lax-Friedrichs splitting for flux approximation
        Input array contains the conservative variables"""
        u = array[1, :] / array[0, :]
        pressure = (self.gamma - 1.0) * (
            array[2, :] - 0.5 * array[1, :] * array[1, :] / array[0, :]
            + self.heat_release * array[3, :])
        sound_speed = np.sqrt(self.gamma*pressure/array[0,:])
        Evals = np.zeros_like(array)
        Evals[0, :] = u - sound_speed
        Evals[1, :] = u
        Evals[2, :] = u + sound_speed
        return np.max(np.abs(Evals), axis=0)

@Equations.register_equations('NonidealReactiveEuler')
class NonidealReactiveEuler(ReactiveEuler):
    def __init__(self, params):
        super().__init__(params)
        if 'cf' not in params.keys():
            raise AttributeError('Parameter cf is not defined for the nonideal reactive Euler equations ')
        self.cf = params['cf']

    def calculate_sources(self, array):
        """ Calculate the right hand side of the equations from the conserved variables"""
        pressure = (self.gamma - 1) * ( array[2, :]
            - 0.5 * array[1, :] * array[1, :] / array[0, :]
            + self.heat_release * array[3, :])

        ### TODO: how to properly pass upstream_cf function and what should its signature be?
        cf_behind_position = upstream_cf(self.cf, frame.position)

        source = np.zeros_like(array)
        source[1, :] = -cf_behind_position * array[1, :] * array[1, :] / array[0, :] / 2
        source[-1, :] = (
            self.rate_const
            * (array[0, :] - array[3, :])
            * np.exp(-self.act_energy * array[0, :] / pressure)
        )
        return source


if __name__=='__main__':
    test_parameters = {'rate_const':1., 'act_energy':26., 'heat_release':50., 'gamma':1.2, 'cf':0.01}
    test_nonideal_eqs = NonidealReactiveEuler(test_parameters)
    print(test_nonideal_eqs)

    test_missed_parameters = {'rate_const':1., 'act_energy':26., 'heat_release':50., 'gamma':1.2}
    try:
        test_nonideal_eqs = NonidealReactiveEuler(test_missed_parameters)
    except Exception as e:
        print(f'Exception occured: {e}')

    print('Testing equations factory')
    factored_equations = Equations.create('NonidealReactiveEuler', test_parameters)
    print(factored_equations)
    try:
        factored_equations = Equations.create('Burgers', test_parameters)
        print(factored_equations)
    except Exception as e:
        print(f'Exception occured: {e}')

    # Test the value of the reaction rate from Henrick et al. (2006)
    print('Testing reaction rate constant')
    calc_rate_const = Equations.create('ReactiveEuler', {'act_energy':25., 'heat_release':50.0, 'gamma':1.2}).calculate_rate_const()
    rate_const_from_Henrick = 35.955584760859722
    try:
        assert np.abs(calc_rate_const - rate_const_from_Henrick) < 1e-13
    except Exception as e:
        print(f'Rate constant is incorrect: {e}')
    else:
        print(f'Rate constant is correct.')
