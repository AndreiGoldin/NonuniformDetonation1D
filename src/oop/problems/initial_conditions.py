import numpy as np
from scipy.integrate import solve_ivp

def initial_znd(nodes, params):
    gamma, Q, act_energy, rate_const = params['gamma'], params['heat_release'],\
                                        params['act_energy'], params['rate_const']
    D_CJ = np.sqrt(gamma + (gamma * gamma - 1.0) * Q / 2.0) + \
            np.sqrt( (gamma * gamma - 1.0) * Q / 2.0)
    gp1 = gamma + 1.
    DCJ2 = D_CJ * D_CJ
    DCJ2p1 = DCJ2 + 1.0
    DCJ2mg = DCJ2 - gamma

    def RHS_lambda(x, y):
        V_lam = gamma / DCJ2  * DCJ2p1 / gp1 * ( 1.0 - DCJ2mg / DCJ2p1 / gamma * (np.sqrt(1 - y)))
        u_lam = 1 / gp1 * DCJ2mg / D_CJ * (1 + np.sqrt(1 - y))
        p_lam = DCJ2p1 / gp1 * ( 1.0 + DCJ2mg / DCJ2p1 * (np.sqrt(1 - y)))
        omega = (1 - y) * np.exp(-act_energy / p_lam / V_lam)
        return rate_const * omega / np.abs(u_lam - D_CJ)

    # Specifying max_step prevents warnings about negative numbers in sqrt
    dx = np.abs(nodes[1]-nodes[0])
    sol = solve_ivp(RHS_lambda,t_span=[0.,-min(nodes)],y0=[0.],t_eval=-nodes[::-1],
                    method='RK45', max_step=dx/2,rtol=1e-13, atol=1e-13)

    lam_x = sol.y[0,::-1]
    V_x = gamma / DCJ2 * DCJ2p1 / gp1 * ( 1.0 - DCJ2mg / DCJ2p1 / gamma * (np.sqrt(1 - lam_x)))
    u_x = 1 / gp1 * DCJ2mg / D_CJ * (1 + np.sqrt(1 - lam_x))
    p_x = DCJ2p1 / gp1 * ( 1.0 + DCJ2mg / DCJ2p1 * (np.sqrt(1 - lam_x)))

    init_array = np.vstack((1 / V_x, u_x, p_x, lam_x))
    return init_array


def initial_znd_lfor(nodes, params):
    znd_part = initial_znd(nodes[nodes<1e-16], params)
    upstream_density = np.ones_like(nodes[nodes>=1e-16])
    upstream_velocity = np.zeros_like(nodes[nodes>=1e-16])
    upstream_pressure = np.ones_like(nodes[nodes>=1e-16])
    upstream_lambda = np.zeros_like(nodes[nodes>=1e-16])
    upstream_part = np.vstack((upstream_density, upstream_velocity,
                                upstream_pressure, upstream_lambda))
    init_array = np.hstack((znd_part,upstream_part))
    return init_array

def initial_znd_lfor_halfwave(nodes, params):
    znd_part = initial_znd(nodes[nodes<1e-16], params)
    upstream_density = np.ones_like(nodes[nodes>=1e-16])
    upstream_velocity = np.zeros_like(nodes[nodes>=1e-16])
    upstream_pressure = np.ones_like(nodes[nodes>=1e-16])
    upstream_lambda = np.zeros_like(nodes[nodes>=1e-16])
    upstream_part = np.vstack((upstream_density, upstream_velocity,
                                upstream_pressure, upstream_lambda))
    init_array = np.hstack((znd_part,upstream_part))
    init_array[0, nodes>50] = 1-np.sin(nodes[nodes>50]-50)
    init_array[0, nodes>50+np.pi] = np.ones_like(nodes[nodes>50+np.pi])
    return init_array
def initial_sine(nodes, params):
    L = max(nodes)-min(nodes)
    return np.sin(2.*np.pi*nodes/L)


def initial_henrick2005(nodes, params):
    return np.sin(np.pi*nodes-np.sin(np.pi*nodes)/np.pi)


def initial_shu_osher(nodes, params):
    init_cond = np.zeros((3, np.size(nodes)))
    init_cond[0,:] = 27./7.*np.ones_like(nodes)*(nodes<-4) + (1.+0.2*np.sin(5*nodes))*(nodes>=-4)
    init_cond[1,:] = 4.*np.sqrt(35)/9*np.ones_like(nodes)*(nodes<-4)
    init_cond[2,:] = 31./3.*np.ones_like(nodes)*(nodes<-4) + np.ones_like(nodes)*(nodes>=-4)
    return init_cond


if __name__=='__main__':
    # Test ZND initial conditions
    rate_const_from_Henrick = 35.955584760859722
    params = {'gamma':1.2, 'act_energy':25.0, 'heat_release':50.0, 'rate_const':rate_const_from_Henrick}
    domain_mesh = np.linspace(-50,0,10001)
    znd = initial_znd(domain_mesh, params)
    # Test that max(lambda) < 1
    assert np.max(znd[3,:]) < 1.0


    import os
    import matplotlib.pyplot as plt
    from pathlib import Path
    Path('test_pics').mkdir(parents=True, exist_ok=True)
    os.chdir('test_pics')
    plt.plot(domain_mesh, znd[0,:])
    plt.savefig('test_znd_density.png')
    plt.close()
    plt.plot(domain_mesh, znd[1,:])
    plt.savefig('test_znd_velocity.png')
    plt.close()
    plt.plot(domain_mesh, znd[2,:])
    plt.savefig('test_znd_pressure.png')
    plt.close()
    plt.plot(domain_mesh, znd[3,:])
    plt.savefig('test_znd_lambda.png')
    plt.close()
