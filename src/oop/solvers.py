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
        self.bound_cond_type = bound_cond_type
        self.space_method_type = space_method_type
        self.equations = Equations.create(equations_type, equation_params)
        self.space_method = SpaceMethod.create(space_method_type, params={'eps':1e-40, 'p':2})
        self.time_method = possible_integrators[time_method_type]
        self.frame = frame_type
        self.init_cond = possible_ic[init_cond_type]
        self.set_bc = lambda x: possible_bc[bound_cond_type](mesh, x)
        self.CFL = CFL

    def set_initial_conditions(self, mesh, params):
        self.space_step = mesh.nodes[1]-mesh.nodes[0]
        united_params = {**params, **self.equations.parameters}
        init_array = self.init_cond(mesh.nodes, united_params)
        init_array = self.set_bc(init_array)
        self.phys_solution = np.copy(init_array)
        # n_row = init_array.shape[0] if np.ndim(init_array) > 1 else 1
        init_array_conserved = self.equations.convert_to_cons_vars(init_array)
        # init_array = np.empty((n_row, mesh.nodes.size))
        # init_array[:, mesh.domain] = init_array_conserved
        init_array_conserved = self.set_bc(init_array_conserved)
        self.solution = np.copy(init_array_conserved)

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
        assert np.allclose(flux, array)
        assert np.sum(rhs) == 0.
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
        # Burgers
        # alpha = np.copy(array)
        # alpha[:, :-1] = np.maximum(np.abs(array[:, :-1]), np.abs(array[:, 1:]))
        # Advection
        # alpha = np.ones_like(array)
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
        self.solution = self.time_method(self.solution, self.calculate_rhs, self.set_bc, self.dt)

    def solve(self, time_limit):
        time = [0.]
        while time[-1] < time_limit:
            mesh = self.mesh
            # assert np.allclose(self.solution[:, mesh.right_ghosts], self.solution[:, mesh.domain[0]+1:mesh.domain[0]+mesh.n_ghosts+1 ])
            # assert np.allclose(self.solution[:, mesh.left_ghosts], self.solution[:, mesh.domain[-1]-mesh.n_ghosts:mesh.domain[-1] ])
            # assert self.solution[:, mesh.domain[0]] == self.solution[:, mesh.domain[-1]]
            # assert np.all(mesh.nodes[mesh.left_ghosts] < 0)
            # assert np.all(mesh.nodes[mesh.right_ghosts] > 1.)

            # assert np.max(self.solution) <= 1.0
            # assert np.min(self.solution) >= -1.0

            dt = self.calculate_dt()
            # assert dt == 8*self.space_step**(5/3)
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
        self.calculate_rhs = self.calculate_rhs_lfweno5m
        self.parameters = params
        # To calculate initial Jacobian norm and then calculate real dt
        # self.set_initial_conditions(mesh, params)
        # self.dt = 0.
        # self.timeintegrate()

    def calculate_dt(self):
        jacobian_norm = self.equations.calculate_jac_norm(self.solution)
        self.dt = self.CFL*self.space_step/np.max(jacobian_norm)
        return self.dt
