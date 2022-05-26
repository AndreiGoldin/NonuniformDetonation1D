# Solver for one bump ahead of the wave
import numpy as np
from reader import Reader
from mesh import Mesh
from solvers import Solver, BurgersSolver, AdvectionSolver
from writer import Writer

reader = Reader()
input = reader.get_input()
solver_type = 'ReactiveEuler'
gamma, Q = 1.2, 50.0
params = {'a':-10,'b':290,'N':6001,'T':41,
          'frame':'LFOR','gamma':gamma,'heat_release':Q}
params = {**params, **input}
E = params['act_energy']
params['init_filename'] = f'init_files/ReactiveEuler_g1.2Q50.0E{E:.1f}_t1000.0'
writer = Writer(params)
# writer.create_folders()

time, time_limit, time_out = [0.], params['T'], 0.
n_image = 0
mesh = Mesh(params['a'], params['b'], params['N'], 3)
params['space_step'] = mesh.nodes[1]-mesh.nodes[0]

solver = Solver.create(solver_type, mesh, params)

solver.set_initial_conditions(mesh, params)
params = {**params, **solver.equations.parameters}
#while time[-1] < time_limit:
#    if time[-1] >= time_out:
#        print(f't = {time[-1]:.2f}')
#        writer.plot_solution(mesh, solver.solution, time[-1], f'image{n_image:03d}', params)
#        time_out += 0.01
#        n_image += 1

#    dt = solver.calculate_dt(solver.solution)
#    solver.dt = dt if time[-1] + dt <= time_limit else time_limit-time[-1]
#    if dt < 1e-15:
#        break

#    solver.timeintegrate()
#    time.append(time[-1] + solver.dt)
#    if np.any(np.isnan(solver.solution)):
#        raise ValueError('Floating point error')
##Last shot
#writer.plot_solution(mesh, solver.solution, time[-1], f'image{n_image:03d}', params)

writer.make_video()
writer.save_solution(solver.equations.convert_to_phys_vars(solver.solution))
