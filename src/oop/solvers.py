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
    def create(cls, solver_type, mesh, params):
        if solver_type not in cls.solvers:
            raise ValueError(f'Unknown solver: {solver_type}. Possible options are {list(cls.solvers.keys())}.')
        return cls.solvers[solver_type](mesh, params)

    def __init__(self, mesh, equations_type='Burgers', equation_params={},
                frame_type='LFOR', init_cond_type='Sine',
                bound_cond_type='Zero', upstream_type=None,
                space_method_type='WENO5M', time_method_type='TVDRK3', CFL=0.8):
        self.equations = Equations.create(equations_type, equation_params)
        self.frame = frame_type
        self.CFL = CFL
        self.bound_cond_type = bound_cond_type
        # self.set_bc = lambda x: possible_bc[bound_cond_type](mesh, x)
        self.set_bc = possible_bc[bound_cond_type](mesh)
        self.space_step = mesh.nodes[1]-mesh.nodes[0]
        self.space_method_type = space_method_type
        self.space_method = SpaceMethod.create(space_method_type, params={'eps':1e-40, 'p':2})
        self.init_cond = possible_ic[init_cond_type]
        # self.time_method = possible_integrators[time_method_type]
        # Compiling with the chosen spatial method and boundary conditions
        self.calculate_rhs = self._create_rhs_func()
        self.time_method = possible_integrators[time_method_type](self.calculate_rhs, self.set_bc)

    def set_initial_conditions(self, mesh, params):
        united_params = {**params, **self.equations.parameters}
        init_array = self.init_cond(mesh.nodes, united_params)
        init_array = self.set_bc(init_array)
        self.phys_solution = np.copy(init_array)
        init_array_conserved = self.equations.convert_to_cons_vars(init_array)
        init_array_conserved = self.set_bc(init_array_conserved)
        self.solution = np.copy(init_array_conserved)

    @staticmethod
    @nb.njit(cache=True)
    def _calculate_rhs_lfweno5m(array, flux_func, source_func,
            spatial_func, jacobian_func, space_step):
        flux = flux_func(array)
        rhs = source_func(array)
        jacobian_norm = jacobian_func(array)
        alpha = np.copy(jacobian_norm)
        alpha[:-1] = np.maximum(np.abs(jacobian_norm[:-1]), np.abs(jacobian_norm[1:]))
        flux_minus = flux - alpha*array
        flux_plus = flux + alpha*array
        flux_approx_minus = spatial_func(flux_minus[:, 5:],
                flux_minus[:, 4:-1], flux_minus[:, 3:-2], flux_minus[:, 2:-3], flux_minus[:,1:-4])
        flux_approx_plus = spatial_func(flux_plus[:, :-5],
                flux_plus[:, 1:-4], flux_plus[:, 2:-3], flux_plus[:, 3:-2], flux_plus[:,4:-1])
        flux_approx = 0.5*(flux_approx_plus + flux_approx_minus)
        flux_derivative = (flux_approx[:, 1:]-flux_approx[:, :-1])/space_step
        rhs[:, 3:-3] -= flux_derivative
        return rhs



    def _create_rhs_func(self):
        """Returns function to be compiled with njit"""
        flux_func = nb.njit(self.equations.calculate_fluxes(), cache=True)
        source_func = nb.njit(self.equations.calculate_sources(), cache=True)
        jacobian_func = nb.njit(self.equations.calculate_jac_norm(), cache=True)
        spatial_func = nb.njit(self.space_method.flux_approx(), cache=True)
        space_step = self.space_step
        rhs_method = self._calculate_rhs_lfweno5m
        def inner(array):
            return rhs_method(array, flux_func, source_func,
                    spatial_func, jacobian_func, space_step)
        return inner


    @abstractmethod
    def calculate_dt(self):
        pass

    def calculate_rhs_upwind(self, array):
        rhs = np.copy(array)
        rhs[:,1:-1] = -array[:, 1:-1]*(array[:, 2:]-array[:, 1:-1])*(array[:, 1:-1]<0.) \
                - array[:, 1:-1]*(array[:, 1:-1]-array[:, :-2])*(array[:, 1:-1]>=0.)
        rhs /= self.space_step
        return rhs

    def calculate_rhs_weno5m(self, array):
        flux = self.equations.calculate_fluxes(array)
        rhs = self.equations.calculate_sources(array)
        # flux_approx[:,0] = f_hat_{1/2}
        # flux_approx[:,-1] = f_hat_{N+1/2}
        flux_approx = self.space_method.flux_approx(flux[:, :-4],
                flux[:, 1:-3], flux[:, 2:-2], flux[:, 3:-1], flux[:,4:])
        # flux_derivative[:,0] = (f_hat_{3/2}-f_hat_{1/2})/dx
        # flux_derivative[:,-1] = (f_hat_{N+1/2}-f_hat_{N-1/2})/dx
        flux_derivative = (flux_approx[:, 1:]-flux_approx[:, :-1])/self.space_step
        rhs[:, 3:-2] -= flux_derivative
        return rhs

    def calculate_rhs_lfweno5m(self, array):
        flux = self.equations.calculate_fluxes(array)
        rhs = self.equations.calculate_sources(array)
        jacobian_norm = self.equations.calculate_jac_norm(array)
        alpha = np.copy(jacobian_norm)
        alpha[:-1] = np.maximum(np.abs(jacobian_norm[:-1]), np.abs(jacobian_norm[1:]))
        flux_minus = flux - alpha*array
        flux_plus = flux + alpha*array
        flux_approx_minus = self.space_method.flux_approx(flux_minus[:, 5:],
                flux_minus[:, 4:-1], flux_minus[:, 3:-2], flux_minus[:, 2:-3], flux_minus[:,1:-4])
        flux_approx_plus = self.space_method.flux_approx(flux_plus[:, :-5],
                flux_plus[:, 1:-4], flux_plus[:, 2:-3], flux_plus[:, 3:-2], flux_plus[:,4:-1])
        flux_approx = 0.5*(flux_approx_plus + flux_approx_minus)
        flux_derivative = (flux_approx[:, 1:]-flux_approx[:, :-1])/self.space_step
        rhs[:, 3:-3] -= flux_derivative
        return rhs

    def timeintegrate(self):
        # self.solution = self.time_method(self.solution, self.calculate_rhs, self.set_bc, self.dt)
        self.solution = self.time_method(self.solution, self.dt)

    def solve(self, time_limit):
        time = [0.]
        while time[-1] < time_limit:
            mesh = self.mesh
            dt = self.calculate_dt()
            self.dt = dt if time[-1] + dt <= time_limit else time_limit-time[-1]
            self.timeintegrate()
            time.append(time[-1] + self.dt)
            if np.any(np.isnan(self.solution)):
                raise ValueError('Floating point error')
        assert np.abs(time_limit-time[-1]) < 1e-20


@Solver.register_solver('Burgers')
class BurgersSolver(Solver):
    def __init__(self, mesh):
        super().__init__(mesh)
        self.calculate_rhs = self.calculate_rhs_lfweno5m

    def calculate_dt(self):
        self.dt = self.CFL * self.space_step / np.max(np.abs(self.solution))
        return self.dt


@Solver.register_solver('Advection')
class AdvectionSolver(Solver):
    def __init__(self, mesh):
        super().__init__(mesh, equations_type='Advection', equation_params={'speed':1.},
                init_cond_type='Henrick2005', bound_cond_type='Periodic',
                # space_method_type='UpstreamCentral',
                space_method_type='WENO5M',
                time_method_type='TVDRK3')
        if self.bound_cond_type == 'Periodic':
            assert self.equations.speed >= 0
            if self.equations.speed >=0:
                self.set_bc = lambda x: possible_bc['PeriodicRight'](mesh, x)
            else:
                self.set_bc = lambda x: possible_bc['PeriodicLeft'](mesh, x)
        elif self.space_method_type in ['WENO5M','UpstreamCentral']:
            self.calculate_rhs = self.calculate_rhs_lfweno5m
        self.mesh = mesh

    def calculate_dt(self):
        self.dt = 8*self.space_step**(5/3)
        return self.dt


@Solver.register_solver('Euler')
class EulerSolver(Solver):
    def __init__(self, mesh, params):
        super().__init__(mesh, equations_type='Euler', equation_params={'gamma':1.4},
                init_cond_type='Shu-Osher', bound_cond_type='Zero_Grad',
                space_method_type='WENO5M', time_method_type='TVDRK3')
        self.calculate_rhs = self.calculate_rhs_lfweno5m
        self.Nt = params['Nt']

    def calculate_dt(self):
        # self.dt = self.space/np.max(self.evals)
        # self.dt = 1.8/400.
        self.dt = 1.8/self.Nt
        return self.dt


@Solver.register_solver('ReactiveEuler')
class ReactiveEulerSolver(Solver):
    def __init__(self, mesh, params):
        super().__init__(mesh, equations_type='ReactiveEuler', equation_params=params,
                init_cond_type='ZND_LFOR_halfwave', bound_cond_type='Zero_Grad',
                space_method_type='WENO5M', time_method_type='TVDRK3')
        # self.calculate_rhs = self.calculate_rhs_lfweno5m
        self.calculate_dt = self._create_dt_func()
        self.parameters = params

    def _calculate_dt(self):
        jacobian_norm = self.equations.calculate_jac_norm(self.solution)
        self.dt = self.CFL*self.space_step/np.max(jacobian_norm)
        return self.dt

    def _create_dt_func(self):
        """Returns function to be compiled with njit"""
        space_step = self.space_step
        CFL = self.CFL
        jacobian_func = nb.njit(self.equations.calculate_jac_norm(), cache=True)
        def inner(array):
            jacobian_norm = jacobian_func(array)
            dt = CFL*space_step/np.max(jacobian_norm)
            return dt
        return nb.njit(inner, cache=True)
