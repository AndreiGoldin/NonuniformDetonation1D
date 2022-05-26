import numpy as np


def density_gradient(array, params):
    return np.gradient(array[0,:], params['space_step'])


def reaction_rate(array, params):
    """ array contains physical solution: rho, u, p, lambda"""
    rate_const = params['rate_const']
    exp_power = -params['act_energy']*array[0,:]/array[2,:]
    return array[0,:]*rate_const*(1. - array[3,:])*np.exp(exp_power)


def reaction_rate_from_cons(array, params):
    """ array contains conservative solution: rho, rho*u, rho*(e + u^2/2), rho*lambda"""
    rate_const = params['rate_const']
    p = pressure(array, params)
    exp_power = -params['act_energy']*array[0,:]/p
    return rate_const*(1. - array[3,:]/array[0,:])*np.exp(exp_power)


def reaction_source_from_cons(array, params):
    """ array contains conservative solution: rho, rho*u, rho*(e + u^2/2), rho*lambda"""
    rate_const = params['rate_const']
    p = pressure(array, params)
    exp_power = -params['act_energy']*array[0,:]/p
    return rate_const*(array[0,:] - array[3,:])*np.exp(exp_power)


def pressure(array, params):
    """ array contains conservative solution: rho, rho*u, rho*(e + u^2/2), rho*lambda"""
    gamma = params['gamma']
    heat_release = params['heat_release']
    pressure = (gamma - 1) * (
        array[2, :]
        - 0.5 * array[1, :] * array[1, :] / array[0, :]
        + heat_release * array[3, :])
    return pressure


def temperature(array, params):
    """ array contains conservative solution: rho, rho*u, rho*(e + u^2/2), rho*lambda"""
    pres = pressure(array, params)
    return pres/array[0,:]

