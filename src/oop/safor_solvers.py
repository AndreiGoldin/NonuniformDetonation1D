# Contains solver classes for various problems in the shock-attached frame of reference
import numpy as np
from problems import *
from  methods import *

class SAFORSolver:
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
                bound_cond_type='Zero', upstream_cond_type='RDE',
                space_method_type='WENO5M', time_method_type='TVDRK3',
                speed_method_type='None', CFL=0.5):
        self.equations = Equations.create(equations_type, equation_params)
        self.frame = frame_type
        self.CFL = CFL
        self.bound_cond_type = bound_cond_type
        # self.set_bc = lambda x: possible_bc[bound_cond_type](mesh, x)
        self.set_bc = possible_bc[bound_cond_type](mesh, equation_params)
        self.space_step = mesh.nodes[1]-mesh.nodes[0]
        self.shock_index = mesh.domain[-1]
        self.space_method_type = space_method_type
        self.space_method = SpaceMethod.create(space_method_type, params={'eps':1e-40, 'p':2})
        self.init_cond = possible_ic[init_cond_type]
        # self.time_method = possible_integrators[time_method_type]
        # Compiling with the chosen spatial method and boundary conditions
        self.calculate_rhs = self._create_rhs_func(self._calculate_rhs_lfweno5m_safor)
        self.calculate_shock_state = possible_uc[upstream_cond_type]

    def set_initial_conditions(self, mesh, params):
        united_params = {**params, **self.equations.parameters}
        init_array = self.init_cond(mesh.nodes, united_params)
        init_array = self.set_bc(init_array, self.shock_state, self.equations.D_CJ)
        self.phys_solution = np.copy(init_array)
        init_array_conserved = self.equations.convert_to_cons_vars(init_array)
        init_array_conserved = self.set_bc(init_array_conserved,
                                           self.shock_state,
                                           self.equations.D_CJ)
        self.solution = np.copy(init_array_conserved)

    def _create_rhs_func(self, rhs_method):
        """Returns function to be compiled with njit"""
        flux_func = nb.njit(self.equations.calculate_fluxes(), cache=True)
        source_func = nb.njit(self.equations.calculate_sources(), cache=True)
        jacobian_func = nb.njit(self.equations.calculate_jac_norm(), cache=True)
        spatial_func = nb.njit(self.space_method.flux_approx(), cache=True)
        space_step = self.space_step
        shock_index = self.shock_index
        def inner(array, shock_speed):
            return rhs_method(array, flux_func, source_func,
                    spatial_func, jacobian_func, space_step,
                    shock_index, shock_speed)
        return inner

    @staticmethod
    @nb.njit(cache=True)
    def _calculate_rhs_lfweno5m_safor(array, flux_func, source_func,
            spatial_func, jacobian_func, space_step, shock_index, shock_speed):
        flux = flux_func(array, shock_speed)
        rhs = source_func(array)
        jacobian_norm = jacobian_func(array, shock_speed)
        # alpha = np.copy(jacobian_norm)
        # alpha[:-1] = np.maximum(np.abs(jacobian_norm[:-1]), np.abs(jacobian_norm[1:]))
        alpha = np.max(jacobian_norm)
        flux_minus = flux - alpha*array
        flux_plus = flux + alpha*array

        flux_approx_minus = spatial_func(flux_minus[:, 5:],
                                         flux_minus[:, 4:-1],
                                         flux_minus[:, 3:-2],
                                         flux_minus[:, 2:-3],
                                         flux_minus[:, 1:-4])

        flux_approx_plus = spatial_func(flux_plus[:,  :-5],
                                        flux_plus[:, 1:-4],
                                        flux_plus[:, 2:-3],
                                        flux_plus[:, 3:-2],
                                        flux_plus[:, 4:-1])

        flux_approx = 0.5*(flux_approx_plus + flux_approx_minus)
        flux_derivative = (flux_approx[:, 1:]-flux_approx[:, :-1])/space_step
        # flux_derivative[:, -1] is at shock
        flux_derivative[:, -3] = ( -2.0 * flux[:, shock_index-5]
                                            + 15.0 * flux[:, shock_index-4]
                                            - 60.0 * flux[:, shock_index-3]
                                            + 20.0 * flux[:, shock_index-2]
                                            + 30.0 * flux[:, shock_index-1]
                                             - 3.0 * flux[:, shock_index]) / 60.0 / space_step
        flux_derivative[:, -2] = ( -flux[:, shock_index-4]
                                       + 6.0 * flux[:, shock_index-3]
                                      - 18.0 * flux[:, shock_index-2]
                                      + 10.0 * flux[:, shock_index-1]
                                       + 3.0 * flux[:, shock_index]) / 12.0 / space_step
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

@SAFORSolver.register_solver('ReactiveEulerSAFOR')
class ReactiveEulerSAFORSolver(SAFORSolver):
    def __init__(self, mesh, params):
        super().__init__(mesh, equations_type='ReactiveEulerSAFOR', equation_params=params,
                init_cond_type='ZND_LFOR', bound_cond_type='Zero_Grad_SAFOR', upstream_cond_type='RDE',
                space_method_type='WENO5M',
                time_method_type='EULER_SAFOR',
                speed_method_type='TVDRK3_SAFOR')
        self.shock_speed = [self.equations.parameters['D_CJ']]
        self.calculate_dt = self._create_dt_func()
        self.calculate_rhs = self._create_rhs_func(self._calculate_rhs_lfweno5m_safor)
        self.calculate_shock_speed_rhs = self._create_speed_func()
        self.time_method = possible_integrators["TVDRK3_SAFOR"](self.calculate_rhs,
                                                                  self.set_bc,
                                                                  self.calculate_shock_speed_rhs)
        # self.set_speed_bc = possible_bc['None'](mesh)
        # self.time_method_speed = possible_integrators['TVDRK3_SPEED'](self.calculate_shock_speed_rhs, self.set_speed_bc)
        self.shock_position = [0.]
        self.ambient = np.array([1.0, 0.0, 1.0, 0.0])
        self.shock_state = (np.array([1.0, 0.0, 1.0, 0.0]), np.array([0.0,0.0,0.0,0.0]))
        self.parameters = params
        self.upstream_params = (params["Arho"], params["krho"], params["Alam"], params["klam"])
        # self.rhs = nb.njit(self.calculate_rhs)
        # self.flux_func = nb.njit(self.equations.calculate_fluxes(), cache=True)

    def _create_dt_func(self):
        """Returns function to be compiled with njit"""
        space_step = self.space_step
        CFL = self.CFL
        shock_speed = self.shock_speed[-1]
        jacobian_func = nb.njit(self.equations.calculate_jac_norm(), cache=True)
        # should have shock_speed as an argument
        def inner(array):
            jacobian_norm = jacobian_func(array, shock_speed)
            dt = CFL*space_step/np.max(jacobian_norm)
            # dt = 8*space_step**(5/3)
            # dt = space_step**2
            return dt
        return nb.njit(inner, cache=True)

    def _create_speed_func(self):
        """Returns function to be compiled with njit"""
        flux_func = nb.njit(self.equations.calculate_fluxes(), cache=True)
        space_step = self.space_step
        shock_index = self.shock_index
        speed_rhs_terms_func = nb.njit(self.speed_rhs_terms, cache=True)
        gamma = self.equations.parameters['gamma']
        def inner(array, shock_state, shock_speed):
            flux = flux_func(array, shock_speed)
            flux_derivative = (-  12.0 * flux[1,shock_index-5]
                               +  75.0 * flux[1,shock_index-4]
                               - 200.0 * flux[1,shock_index-3]
                               + 300.0 * flux[1,shock_index-2]
                               - 300.0 * flux[1,shock_index-1]
                               + 137.0 * flux[1,shock_index]) / (60.0 * space_step)
            dspeed_dm, dmomentum_dxi = speed_rhs_terms_func(shock_speed, shock_state, gamma)
            return -dspeed_dm * (flux_derivative + shock_speed * dmomentum_dxi)
        return inner

    @staticmethod
    def speed_rhs_terms(shock_speed, shock_state, gamma):
        shock_state, shock_state_der = shock_state
        rho_a, u_a, p_a, lambda_a = (
                shock_state[0],
                shock_state[1],
                shock_state[2],
                shock_state[3],
            )

        drho_dx, du_dxi, dp_dxi, dlambda_dxi = (
                shock_state_der[0],
                shock_state_der[1],
                shock_state_der[2],
                shock_state_der[3],
            )

        speed_dif = shock_speed - u_a
        nom = ( rho_a * speed_dif * (
                gamma * (rho_a * u_a * speed_dif - 2.0 * p_a)
                + rho_a * (2.0 * shock_speed**2 - 3.0 * shock_speed * u_a + u_a**2)))
        denom = gamma * (2.0 * p_a + rho_a * speed_dif**2) - rho_a * speed_dif**2
        nom_over_denom_sq = nom / denom**2
        # The derivative of the momentum at the shock \rho*u|_s w.r.t. the shock velocity D
        nom_der = nom / speed_dif + rho_a * speed_dif * (
            gamma * rho_a * u_a + rho_a * (4.0 * shock_speed - 3.0 * u_a))
        denom_der = 2.0 * rho_a * (gamma - 1.0) * speed_dif
        dspeed_dm = 1.0 / (nom_der / denom - denom_der * nom_over_denom_sq)
        # The derivative of the momentum at the shock \rho*u|_s w.r.t. the laboratory coordinate \xi
        dm_dxi = 0
        if drho_dx >= 1e-8:
            nom_der = nom / rho_a + rho_a * speed_dif * (
                gamma * (u_a * speed_dif) +
                (2.0 * shock_speed**2 - 3.0 * shock_speed * u_a + u_a**2))
            denom_der = (gamma - 1.0) * speed_dif**2
            dm_drho = nom_der / denom - denom_der * nom_over_denom_sq
            dm_dxi += dm_drho * drho_dx
        if du_dxi >= 1e-8:
            nom_der = -nom / speed_dif + rho_a**2 * speed_dif * (
                (gamma - 3.0) * shock_speed -  2.0 * u_a * (gamma - 1.0))
            denom_der = 2.0 * (1.0 - gamma) * rho_a * speed_dif
            dm_du = nom_der / denom - denom_der * nom_over_denom_sq
            dm_dxi += dm_du * du_dxi
        if dp_dxi >= 1e-8:
            nom_der = -2.0 * rho_a * speed_dif * gamma
            denom_der = 2.0 * gamma
            dm_dp = nom_der / denom - denom_der * nom_over_denom_sq
            dm_dxi += dm_dp * dp_dxi
        return dspeed_dm, dm_dxi

    # def calculate_shock_speed(self):
    #     self.shock_state = self.calculate_shock_state(self.shock_position[-1], self.upstream_params)
    #     shock_speed_new = self.time_method_speed(self.solution, self.dt, self.shock_state, self.shock_speed[-1])
    #     self.shock_position.append(self.shock_position[-1] + self.dt*self.shock_speed[-1])
    #     self.shock_speed.append(shock_speed_new)

    def timeintegrate(self):
        self.solution, new_speed = self.time_method(self.solution, self.dt, self.shock_state, self.shock_speed[-1])
        self.phys_solution = self.equations.convert_to_phys_vars(self.solution)
        self.shock_speed.append(new_speed)
        self.shock_position.append(self.shock_position[-1] + self.dt*self.shock_speed[-1])
        self.shock_state = self.calculate_shock_state(self.shock_position[-1], self.upstream_params)


@SAFORSolver.register_solver('AdvectionSAFOR')
class AdvectionSolver(SAFORSolver):
    def __init__(self, mesh, params):
        super().__init__(mesh, equations_type='AdvectionSAFOR', equation_params={'speed':1.},
                init_cond_type='Sine', bound_cond_type='PeriodicRight',
                # space_method_type='UpstreamCentral',
                space_method_type='WENO5M',
                time_method_type='EULER_SAFOR')
        if self.bound_cond_type == 'Periodic':
            assert self.equations.speed >= 0
            if self.equations.speed >=0:
                self.set_bc = lambda x,y,z: possible_bc['PeriodicRight'](mesh, x)
            else:
                self.set_bc = lambda x,y,z: possible_bc['PeriodicLeft'](mesh, x)
        elif self.space_method_type in ['WENO5M','UpstreamCentral']:
            # self.calculate_rhs = self.calculate_rhs_lfweno5m
            self.calculate_rhs = self._create_rhs_func(self._calculate_rhs_lfweno5m_safor)
        self.mesh = mesh
        self.shock_state=0.0
        self.shock_speed=[0.5]
        self.calculate_shock_speed_rhs = lambda x,y,z : 1.0
        self.time_method = possible_integrators["TVDRK3_SAFOR"](self.calculate_rhs,
                                                                  self.set_bc,
                                                                  self.calculate_shock_speed_rhs)
    def calculate_dt(self, array):
        self.dt = 8*self.space_step**(5/3)
        return self.dt

    def timeintegrate(self):
        self.solution, new_speed = self.time_method(self.solution, self.dt, self.shock_state, self.shock_speed[-1])
        self.phys_solution = self.equations.convert_to_phys_vars(self.solution)
        # self.shock_speed.append(1.0)


@SAFORSolver.register_solver('BurgersSAFOR')
class BurgersSolver(SAFORSolver):
    def __init__(self, mesh, params):
        super().__init__(mesh, equations_type='BurgersSAFOR', equation_params={'speed':1.},
                init_cond_type='Sine', bound_cond_type='Zero_Grad_SAFOR',
                # space_method_type='UpstreamCentral',
                space_method_type='WENO5M',
                time_method_type='EULER_SAFOR')
        if self.bound_cond_type == 'Periodic':
            assert self.equations.speed >= 0
            if self.equations.speed >=0:
                self.set_bc = lambda x,y,z: possible_bc['PeriodicRight'](mesh, x)
            else:
                self.set_bc = lambda x,y,z: possible_bc['PeriodicLeft'](mesh, x)
        elif self.space_method_type in ['WENO5M','UpstreamCentral']:
            # self.calculate_rhs = self.calculate_rhs_lfweno5m
            self.calculate_rhs = self._create_rhs_func(self._calculate_rhs_lfweno5m_safor)
        self.mesh = mesh
        self.shock_state=0.0
        self.shock_speed=[1.0]
        self.calculate_shock_speed_rhs = lambda x,y,z : 0.0
        self.time_method = possible_integrators["TVDRK3_SAFOR"](self.calculate_rhs,
                                                                  self.set_bc,
                                                                  self.calculate_shock_speed_rhs)
    def calculate_dt(self, array):
        self.dt = 8*self.space_step**(5/3)
        return self.dt

    def timeintegrate(self):
        self.solution, new_speed = self.time_method(self.solution, self.dt, self.shock_state, self.shock_speed[-1])
        self.phys_solution = self.equations.convert_to_phys_vars(self.solution)
        # self.shock_speed.append(1.0)

