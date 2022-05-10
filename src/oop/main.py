# Solver for hyperbolic problems in 1D
import numpy as np
from reader import Reader
from mesh import Mesh
from solvers import Solver, BurgersSolver, AdvectionSolver
from writer import Writer

# Reader.get_input
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
solver_type = 'ReactiveEuler'
gamma, Q, E = 1.2, 50.0, 25.0
params = {'a':-10,'b':100,'N':20001,'T':15.0,'frame':'LFOR',
        'gamma':gamma,'heat_release':Q,'act_energy':E}

time, time_limit, time_out = [0.], params['T'], 0.
n_image = 0
mesh = Mesh(params['a'], params['b'], params['N'], 3)

solver = Solver.create(solver_type, mesh, params)

solver.set_initial_conditions(mesh, params)
while time[-1] < time_limit:
    if time[-1] >= time_out:
        print(f't = {time[-1]:.2f}')
        Writer.plot_solution(mesh, solver.solution, time[-1], f'image{n_image:03d}')
        time_out += 0.01
        n_image += 1
    dt = solver.calculate_dt()
    solver.dt = dt if time[-1] + dt <= time_limit else time_limit-time[-1]
    if dt < 1e-15:
        break
    solver.timeintegrate()
    time.append(time[-1] + solver.dt)
    if np.any(np.isnan(solver.solution)):
        raise ValueError('Floating point error')

#Last shot
Writer.plot_solution(mesh, solver.solution, time[-1], f'image{n_image:03d}')
Writer.make_video()
# filename = f"test_data/Euler_Nx{params['N']}Nt12800"
# Writer.save_solution(filename,solver.equations.convert_to_phys_vars(solver.solution))

if params['frame'] == 'SAFOR':
    # Writer.write_speed(solver.shock_speed)
    pass
