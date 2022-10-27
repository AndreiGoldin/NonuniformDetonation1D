# Solver for hyperbolic problems in 1D
import time as timer
import numpy as np
from reader import Reader
from mesh import Mesh
from solvers import Solver, BurgersSolver, AdvectionSolver
from safor_solvers import SAFORSolver
from writer import Writer

# Reader.get_input()
Writer.create_folders()
# Burgers
# params = {'a':0., 'b':2*np.pi, 'N': 1001, 'T':10., 'frame':'LFOR'}
# Henrick2005: Advection
# params = {'a':-1., 'b':1. , 'N': 51, 'T':2., 'frame':'LFOR'}
# Henrick2005: Euler
# solver_type = 'Euler'
# params = {'a':-5, 'b':5, 'N':12801, 'T':1.8, 'frame':'LFOR', 'Nt':12800}
# params = {'a':-5, 'b':5, 'N':401, 'T':1.8, 'frame':'LFOR', 'Nt':400}
# Henrick2006: Reactive Euler
solver_type = 'ReactiveEulerSAFOR'
# solver_type = 'AdvectionSAFOR'
# solver_type = 'BurgersSAFOR'
gamma, Q, E = 1.2, 50.0, 26.0
# params = {'a':0.,'b':1.,'N':101,'T':5.0,'frame':'SAFOR',
params = {'a':-30.,'b':0.,'N':301,'T':500.0,'frame':'SAFOR',
        'gamma':gamma,'heat_release':Q,'act_energy':E,
        'upstream_cond_type':'RDE', 'Arho':0.0, 'krho':0.0, 'Alam':0.0, 'klam':0.0}
SAFOR = params['frame'] == 'SAFOR'

time, time_limit, time_out = [0.], params['T'], 0.
n_image = 0
mesh = Mesh(params['a'], params['b'], params['N'], 3)

# solver = Solver.create(solver_type, mesh, params)
solver = SAFORSolver.create(solver_type, mesh, params)

solver.set_initial_conditions(mesh, params)
#
timer_start = timer.perf_counter()
while time[-1] < time_limit:
    if time[-1] >= time_out:
        print(f't = {time[-1]:.2f}')
        # Writer.plot_solution(mesh, solver.phys_solution, time[-1], f'image{n_image:03d}')
        time_out += 0.01
        n_image += 1

    dt = solver.calculate_dt(solver.solution)
    solver.dt = dt if time[-1] + dt <= time_limit else time_limit-time[-1]
    if dt < 1e-15:
        break

    solver.timeintegrate()
    # if SAFOR:
    #     solver.calculate_shock_speed()

    time.append(time[-1] + solver.dt)
    if np.any(np.isnan(solver.solution)):
        # raise ValueError('Floating point error')
        print('Floating point error')
        break

#Last shot
# Writer.plot_solution(mesh, solver.solution, time[-1], f'image{n_image:03d}')
Writer.plot_solution(mesh, solver.phys_solution, time[-1], f'image{n_image:03d}')
wall_time = timer.perf_counter() - timer_start
print(f'Walltime: {wall_time:.2f} s')

# Writer.make_video()
# filename = f"test_data/Euler_Nx{params['N']}Nt12800"
# Writer.save_solution(filename,solver.equations.convert_to_phys_vars(solver.solution))

if params['frame'] == 'SAFOR':
    Writer.plot_speed(time, solver.shock_speed, 'test_speed')
