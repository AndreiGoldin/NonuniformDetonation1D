# Solver for hyperbolic problems in 1D
# set_initial()
# toConserved
import numpy as np
from reader import Reader
from mesh import Mesh
from solvers import Solver, BurgersSolver
from writer import Writer

# Reader.get_input
params = {'L':2*np.pi, 'N': 101, 'T':10., 'frame':'LFOR'}

time, time_limit, time_out = [0.], params['T'], 0.
n_image = 0
mesh = Mesh(0., params['L'], params['N'], 3)

# solver = Solver.create(params)
solver = BurgersSolver(mesh)
solver.set_initial_conditions(mesh)
dt = solver.calculate_dt()

while time[-1] < time_limit:
    if time[-1] >= time_out:
        print(f't = {time[-1]:.2f}')
        Writer.plot_solution(mesh, solver.solution, f'image{n_image:03d}')
        time_out += 0.01
        n_image += 1
    solver.timeintegrate()
    time.append(time[-1] + dt)
    dt = solver.calculate_dt()

Writer.make_video()

if params['frame'] == 'SAFOR':
    # Writer.write_speed(solver.shock_speed)
    pass
