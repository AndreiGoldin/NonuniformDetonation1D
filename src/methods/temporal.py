# Contains methods for time integration
import numpy as np
import numba as nb


def euler(array, rhs, set_bc, dt):
    update = array + dt*rhs(array)
    update = set_bc(update)
    return update


def euler_safor(rhs, set_bc, rhs_speed):
    """ Closure for numba version of euler"""
    set_bc = nb.njit(set_bc, cache=True)
    rhs = nb.njit(rhs)
    rhs_speed = nb.njit(rhs_speed, cache=True)
    def inner(array, dt, ambient, speed):
        update = array + dt*rhs(array, speed)
        upd_speed = speed + dt*rhs_speed(array, ambient, speed)
        update = set_bc(update, ambient, upd_speed)
        return update, upd_speed
    return nb.njit(inner)


def _tvd_rk3(array, rhs, set_bc, dt):
    update = array + dt*rhs(array)
    update = set_bc(update)
    update1 = 3./4.*array + 1./4.*update + 1./4.*dt*rhs(update)
    update1 = set_bc(update1)
    update2 = 1./3.*array + 2./3.*update1 + 2./3.*dt*rhs(update1)
    update2 = set_bc(update2)
    return update2


def tvd_rk3(rhs, set_bc):
    """ Closure for numba version of tvd_rk3"""
    set_bc = nb.njit(set_bc, cache=True)
    rhs = nb.njit(rhs)
    def inner(array, dt):
        update = array + dt*rhs(array)
        update = set_bc(update)
        update1 = 3./4.*array + 1./4.*update + 1./4.*dt*rhs(update)
        update1 = set_bc(update1)
        update2 = 1./3.*array + 2./3.*update1 + 2./3.*dt*rhs(update1)
        update2 = set_bc(update2)
        return update2
    return nb.njit(inner)


def tvd_rk3_safor(rhs, set_bc, rhs_speed):
    """ Closure for numba version of tvd_rk3"""
    set_bc = nb.njit(set_bc, cache=True)
    rhs = nb.njit(rhs)
    rhs_speed = nb.njit(rhs_speed, cache=True)
    def inner(array, dt, ambient, speed):
        update = array + dt*rhs(array, speed)
        upd_speed = speed + dt*rhs_speed(array, ambient, speed)
        update = set_bc(update, ambient, upd_speed)

        update1 = 3./4.*array + 1./4.*update + 1./4.*dt*rhs(update, upd_speed)
        upd1_speed = 3./4.*speed + 1./4.*upd_speed + 1./4.*dt*rhs_speed(update, ambient, upd_speed)
        update1 = set_bc(update1, ambient, upd1_speed)

        update2 = 1./3.*array + 2./3.*update1 + 2./3.*dt*rhs(update1, upd1_speed)
        upd2_speed = 1./3.*speed + 2./3.*upd1_speed + 2./3.*dt*rhs_speed(update1, ambient, upd1_speed)
        update2 = set_bc(update2, ambient, upd2_speed)
        return update2, upd2_speed
    return nb.njit(inner)


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
