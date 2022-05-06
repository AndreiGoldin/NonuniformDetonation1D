# Contains methods for time integration
import numpy as np

class TimeMethod:
    pass


def explicit_euler(array, rhs, set_bc, dt):
    update = np.copy(array)
    update = array + dt*rhs(array)
    update = set_bc(update)
    return update
