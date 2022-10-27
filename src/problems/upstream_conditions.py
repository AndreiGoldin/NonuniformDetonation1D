# Contains functions for various initial, boundary and upstream conditions
from scipy import interpolate
import numpy as np


def upstream_periodic_rho(x, mean=1., amp=0., wn=0.):
    """Calculates density at lab coordinate x"""
    u_a, p_a, lambda_a = 0.0, 1.0, 0.0
    u_der, p_der,lambda_der = 0.0, 0.0, 0.0
    if wn == 0. or amp == 0.:
        rho_a, rho_der = mean, 0.0
        return np.array((rho_a, u_a, p_a, lambda_a)), np.array((rho_der, u_der, p_der, lambda_der))
    if x >= 1 / wn:
        # For temperature
        rho_a = max( ( 0, 1.0 / ( 1.0 + amp * ( np.cos(2.0 * np.pi * wn * x) - 1.0)),))
        rho_der = ( 0 if rho_a == 0 else 2.0 * np.pi * wn * amp * np.sin(2.0 * np.pi * wn * x) * rho_a**2)
    else:
        x_for_const = np.linspace(0.0, 1.0 / wn / 2.0)
        y_for_const = mean * np.ones_like(x_for_const)
        x_for_periodic = np.linspace( 1.0 /wn , 2.0 /wn )
        y_for_periodic = np.array( 1.0 / ( 1.0 + amp * ( np.cos( 2.0 * np.pi * wn * x_for_periodic) - 1.0)))
        y_for_periodic = np.where( y_for_periodic >= 0, y_for_periodic, 0)
        x_for_spline = np.hstack([x_for_const, x_for_periodic])
        y_for_spline = np.hstack([y_for_const, y_for_periodic])
        tck = interpolate.splrep(x_for_spline, y_for_spline)
        rho_a = max((0, interpolate.splev(x, tck, der=0)))
        rho_der = (0 if rho_a == 0 else interpolate.splev(x, tck, der=1))
    return np.array((rho_a, u_a, p_a, lambda_a)), np.array((rho_der, u_der, p_der, lambda_der))


def upstream_periodic_lambda(x, minimum=0., amp=0., wn=0.):
    """Calculates lambda at lab coordinate x"""
    rho_a, u_a, p_a = 1.0, 0.0, 1.0
    rho_der, u_der, p_der = 0.0, 0.0, 0.0
    if wn == 0. or amp == 0.:
        lambda_a, lambda_der = minimum, 0.
        return np.array((rho_a, u_a, p_a, lambda_a)), np.array((rho_der, u_der, p_der, lambda_der))
    if x >= 1 / wn:
        # For temperature
        lambda_a = max( ( 0, 1.0 - np.exp( amp * (np.cos(2 * np.pi * wn * x) - 1)),))
        lambda_der = ( 0 if lambda_a == 0 else 2.0 * np.pi * wn * amp * np.sin(2.0 * np.pi * wn * x) * (1.0 - lambda_a))
    else:
        x_for_const = np.linspace(0.0, 1.0 / wn / 2.0)
        y_for_const = minimum * np.ones_like(x_for_const)
        x_for_periodic = np.linspace( 1.0 / wn, 2.0 / wn)
        y_for_periodic = np.array( 1.0 - np.exp( amp * ( np.cos( 2 * np.pi * wn * x_for_periodic)-1)))
        y_for_periodic = np.where( y_for_periodic >= 0, y_for_periodic, 0)
        x_for_spline = np.hstack([x_for_const, x_for_periodic])
        y_for_spline = np.hstack([y_for_const, y_for_periodic])
        tck = interpolate.splrep(x_for_spline, y_for_spline)
        lambda_a = max((0, interpolate.splev(x, tck, der=0)))
        lambda_der = (0 if lambda_a == 0 else interpolate.splev(x, tck, der=1))
    return np.array((rho_a, u_a, p_a, lambda_a)), np.array((rho_der, u_der, p_der, lambda_der))


def upstream_rde(x,params):
    """Conditions similar to perturbations in the rotating detonation engines"""
    u_a, u_der = 0.0, 0.0
    p_a, p_der = 1.0, 0.0

    rho_amp, rho_wn, lambda_amp, lambda_wn = params
    vals, ders = upstream_periodic_rho(x,amp=rho_amp,wn=rho_wn)
    rho_a, rho_der = vals[0], ders[0]
    vals, ders = upstream_periodic_lambda(x,amp=lambda_amp, wn=lambda_wn)
    lambda_a, lambda_der = vals[-1], ders[-1]
    return np.array((rho_a, u_a, p_a, lambda_a)), np.array((rho_der, u_der, p_der, lambda_der))


def upstream_uniform(x,params):
    """The simplest constant state"""
    rho_a, rho_der = 1.0, 0.0
    u_a, u_der = 0.0, 0.0
    p_a, p_der = 1.0, 0.0
    lambda_a, lambda_der = 0.0, 0.0
    return np.array((rho_a, u_a, p_a, lambda_a)), np.array((rho_der, u_der, p_der, lambda_der))


if __name__=='__main__':
    # Tests
    import matplotlib.pyplot as plt
    params = (0.01, 1., 0.5, 0.5)
    x_mesh = np.linspace(0,10,500)
    x_rho, x_u, x_p, x_lam  = [], [], [], []
    x_rho_der, x_u_der, x_p_der, x_lam_der  = [], [], [], []
    for i, x in enumerate(x_mesh):
        vals, ders = upstream_RDE(x, params)
        x_rho.append(vals[0])
        x_u.append(vals[1])
        x_p.append(vals[2])
        x_lam.append(vals[3])
        x_rho_der.append(ders[0])
        x_u_der.append(ders[1])
        x_p_der.append(ders[2])
        x_lam_der.append(ders[3])

    import os
    import matplotlib.pyplot as plt
    from pathlib import Path
    Path('test_pics').mkdir(parents=True, exist_ok=True)
    os.chdir('test_pics')
    plt.plot(x_mesh, x_rho)
    plt.savefig('test_periodic_RDE_rho.png')
    plt.close()
    plt.plot(x_mesh, x_u)
    plt.savefig('test_periodic_RDE_u.png')
    plt.close()
    plt.plot(x_mesh, x_p)
    plt.savefig('test_periodic_RDE_p.png')
    plt.close()
    plt.plot(x_mesh, x_lam)
    plt.savefig('test_periodic_RDE_lam.png')
    plt.close()
    plt.plot(x_mesh, x_rho_der)
    plt.savefig('test_periodic_RDE_rho_der.png')
    plt.close()
    plt.plot(x_mesh, x_u_der)
    plt.savefig('test_periodic_RDE_u_der.png')
    plt.close()
    plt.plot(x_mesh, x_p_der)
    plt.savefig('test_periodic_RDE_p_der.png')
    plt.close()
    plt.plot(x_mesh, x_lam_der)
    plt.savefig('test_periodic_RDE_lam_der.png')
    plt.close()
