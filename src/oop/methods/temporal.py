# Contains methods for time integration
import numpy as np

class TimeMethod:
    pass


def explicit_euler(array, rhs, set_bc, dt):
    update = array + dt*rhs(array)
    update = set_bc(update)
    return update


def tvd_rk3(array, rhs, set_bc, dt):
    update = array + dt*rhs(array)
    update = set_bc(update)
    update1 = 3./4.*array + 1./4.*update + 1./4.*dt*rhs(update)
    update1 = set_bc(update1)
    update2 = 1./3.*array + 2./3.*update1 + 2./3.*dt*rhs(update1)
    update2 = set_bc(update2)
    return update2


def tvd_rk5(array, rhs, set_bc, dt):
    update = array + dt*rhs(array)
    update = set_bc(update)
    update2 = array + dt*(0.25*rhs(array)+0.25*rhs(update))
    update2 = set_bc(update2)
    update3 = array + dt*(2046./15625.*rhs(array) - 454./15625.*rhs(update)
            + 1533./15625.*rhs(update2))
    update3 = set_bc(update3)
    update4 = array + dt*(-739./5625.*rhs(array) + 511./5625.*rhs(update)
            - 566./16875.*rhs(update2) + 20./27.*rhs(update3))
    update4 = set_bc(update4)
    update5 = array + dt*(11822./21857.*rhs(array) - 6928./21857.*rhs(update)
            - 4269./21857.*rhs(update2) -4./7.*rhs(update3) + 54./35.*rhs(update4))
    update5 = set_bc(update5)
    final = array + dt*(1./24.*rhs(array) + 125./336.*rhs(update3)
            + 27./56.*rhs(update4) + 5./48.*rhs(update5))
    final = set_bc(final)
    return final
