# Contains solver classes for various problems
from abc import abstractmethod
import numpy as np
from problems import *
from  methods import *


class Solver:
    solvers = {}

    @classmethod
    def register_solver(cls, solver_type):
        def decorator(subclass):
            cls.solvers[solver_type] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, solver_type, params):
        if solver_type not in cls.solvers:
            raise ValueError(f'Unknown solver: {solver_type}. Possible options are {list(cls.solvers.keys())}.')
        return cls.solvers[solver_type](params)

    @abstractmethod
    def calculate_dt(self):
        pass

    @abstractmethod
    def timeintegrate(self):
        pass


@Solver.register_solver('Burgers')
class BurgersSolver(Solver):
    def __init__(self, mesh, frame_type='LFOR', init_cond_type='Sine',
                bound_cond_type='Zero', upstream_type=None,
                space_method='WENO5M', time_method='ExplicitEuler', CFL=0.8):
        self.equations = Equations.create('Burgers', params={})
        self.space_method = SpaceMethod.create(space_method, params={'eps':1e-40})
        self.time_method = possible_integrators[time_method]
        self.frame = frame_type
        self.init_cond = possible_ic[init_cond_type]
        self.set_bc = lambda x: possible_bc[bound_cond_type](mesh, x)
        self.CFL = CFL

    def set_initial_conditions(self, mesh):
        init_array = self.init_cond(mesh.nodes[mesh.domain], params={})
        n_row = init_array.shape[0] if np.ndim(init_array) > 1 else 1
        init_array_conserved = self.equations.convert_to_cons_vars(init_array)
        init_array = np.empty((n_row, mesh.nodes.size))
        init_array[:, mesh.domain] = init_array_conserved
        self.set_bc(init_array)
        self.space_step = mesh.nodes[1]-mesh.nodes[0]
        self.solution = init_array

    def calculate_dt(self):
        self.dt = self.CFL * self.space_step / np.max(np.abs(self.solution))
        return self.dt

    # def calculate_rhs(self, array):
    #     # 2:-3 = domain = nog:-nog
    #     flux = self.equations.calculate_fluxes(array)
    #     source = self.equations.calculate_sources(array)
    #     rhs = source
    #     alpha = np.maximum(array[:, :-1], array[:, 1:])
    #     flux_minus = flux[:, :-1] - alpha*array[:, :-1]
    #     flux_plus = flux[:, :-1] + alpha*array[:, :-1]
    #     flux_approx_minus = self.space_method.flux_approx(flux_minus[:, 5:],
    #             flux_minus[:, 4:-1], flux_minus[:, 3:-2], flux_minus[:, 2:-3], flux_minus[:,1:-4])
    #     flux_approx_plus = self.space_method.flux_approx(flux_plus[:, :-5],
    #             flux_plus[:, 1:-4], flux_plus[:, 2:-3], flux_plus[:, 3:-2], flux_plus[:,4:-1])
    #     flux_approx = 0.5*(flux_approx_plus + flux_approx_minus)
    #     flux_derivative = (flux_approx[:, 1:]-flux_approx[:, :-1])/self.space_step
    #     rhs[:, 3:-4] -= flux_derivative
    #     return rhs

    def calculate_rhs(self, array):
        rhs = np.copy(array)
        rhs[:,1:-1] = - array[:, 1:-1]*(array[:, 2:]-array[:, 1:-1])/self.space_step*(array[:, 1:-1]<0.) \
                - array[:, 1:-1]*(array[:, 1:-1]-array[:, :-2])/self.space_step*(array[:, 1:-1]>=0.)
        return rhs

    def timeintegrate(self):
        self.solution = self.time_method(self.solution, self.calculate_rhs, self.set_bc, self.dt)






