# DESCRIPTION
This package contains a Python solver for 1D hyperbolic equations and systems. Main purpose is to
implement the algorithm for the reactive Euler equations in the shock-attached frame of reference (SAFOR)
that is presented in [[1]](#1). However, in order to test different time and spatial discretization
methods, the advection, Burgers, and Euler equations are also included as well as the possibility
to work in the laboratory frame of reference. To enhance performance, [`numba` package](https://numba.pydata.org/) is used.

# INSTALLATION
To use the package, you need Python with version >=3.6 and I recommend to use a separate
virtual environment for simulations, for example, via [`venv` package](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).
Please execute following commands in the terminal.
```
git clone https://github.com/AndreiGoldin/NonuniformDetonation1D.git
cd NonuniformDetonation1D
python3 -m venv solver-env
source solver-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
To test the installation run
```
cd examples
python3 advection.py
```
or other examples.
If you want to have animations of the evolution, you need to install [`ffmpeg`](https://ffmpeg.org/download.html).
NB: When writing video is enabled, the solver works slower due to frame saving.

# USAGE
## Configuration
The solver uses `config.yaml` file to prepare and run simulation. This file contains
several sections for various part of the simulation.
### Problem
The `type` key allows to choose the equation or system to be solved while
`parameters` should specify names and values of the parameters for the chosen problem.
Currently the following equations are included (possible opitons for `type` value and `parameters` list)
* `Advection`
```math
u_t + au_x = 0,
```
where $a$ is the `speed` of the wave.
* `Burgers`
```math
u_t + uu_x = 0
```
with no additional parameters.
* `Euler`
```math
\begin{align}
& \frac{\partial\rho}{\partial t} + \frac{\partial}{\partial x}\left( \rho u\right) = 0, \\
& \frac{\partial}{\partial t} \left( \rho u\right) + \frac{\partial}{\partial x}\left( \rho u^2+ p\right) = 0, \\
& \frac{\partial}{\partial t} E  + \frac{\partial}{\partial x} \left( u E + u p\right)= 0, \\
\end{align}
```
where $\rho$, $u$, and $p$ are the gas density, velocity, and pressure respectively.
The total energy of gas $E = p/(\gamma - 1) + \rho u^2 / 2$ contains one parameter $\gamma$ (`gamma`).
* `ReactiveEuler`
```math
\begin{align}
& \frac{\partial\rho}{\partial t} + \frac{\partial}{\partial x}\left( \rho u\right) = 0, \\
& \frac{\partial}{\partial t} \left( \rho u\right) + \frac{\partial}{\partial x}\left( \rho u^2+ p\right) = 0, \\
& \frac{\partial}{\partial t} \left(\rho \left(e + \frac 12 u^2\right)\right) + \frac{\partial}{\partial x} \left(\rho u \left(e + \frac 12 u^2 \right) + u p\right)= 0, \\
& \frac{\partial}{\partial t} \left( \rho \lambda \right)  + \frac{\partial}{\partial x}\left( \rho u\lambda \right) = K \rho (1-\lambda) \exp\left(-\frac{\rho E}{p}\right), \\
\end{align}
```
where $\rho$, $u$, and $p$ have the same meaning as for the Euler equations above.
The specific internal energy $e$ is $p\rho^{-1}/(\gamma -1) - \lambda Q$ with `heat release` $Q$.
The fourth equation describes the evolution of the concentration $\lambda$ of the products
for one-step chemical reaction. Its kinetics obeys Arrhenius law with `activation energy` $E$.
The pre-exponential factor `K` is scaled in such a way that $\lambda(-1) = 1/2$ for the
steady detonation wave with the shock located at $x=0$.

The `frame` can be either `Laboratory` for laboratory frame of reference or
`Shock` for the shock-attached frame of reference for the reactive Euler equations.

Initial conditions are chosen with `initial` key. Currently possible options are
* `Sine`: $u_0(x) = \sin (2\pi x)$
* `Henrick2005` (see [[2]](#2)): $u_0(x) = \sin(\pi x - \sin(\pi x)/\pi) $
* `Shu-Osher` (see [[2]](#2)): Shock entropy wave interaction problem for the Euler equations.
* `ZND` (see [[4]](#4)): The detonation wave with $\gamma$, $E$, and $Q$ specified in `parameters`
and initial velocity
```math
D_{CJ} = \sqrt{\gamma + \frac 12 \left( \gamma^2 - 1 \right) Q} + \sqrt{\frac 12 \left( \gamma^2 - 1 \right) Q}.
```
Initial shock position is at $x=0$.

The key `boundary` is responsible for the boundary conditions that can be chosen from
- `Zero`
- `Periodic`
- `Zero gradient`


### Domain
Here you should specify the boundaries of the solution interval, number of nodes
and the final simulation time.

### Methods
Possible methods for spatial discretisation:
* `Upwind` of the 1st order.
* `UpstreamCentral` of the 5th order (see [[2]](#2)).
* `WENO5M`: mapped weighted essentially non-oscillatory scheme of the 5th order (see [[1]](#1) and [[2]](#2)).

Possible methods for time integration:
* `Euler` of the first order.
* `TVDRK3`: total variation diminishing Runge-Kutta of the 3rd order (see [[2]](#2)).
* `TVDRK5`: total variation diminishing Runge-Kutta of the 5th order (see [[1]](#1)).

### Upstream
This section is used to prescribe the `type` and `parameters` for upstream conditions in the case of the
shock-attached frame of reference.
* `Uniform`: constant conditions
* `RDE`: sinusoidal perturbations of the temperature and fuel concentration ahead of the wave
similar to the flow inside the rotating detonation engine (see details below). List of parameters
includes `density amplitude`, `density wavenumber`, `lambda amplitude`, and `lambda wavenumber`.

### Callbacks
Only `yes` or `no` options are available for each key. Saved files can be found in `results` folder.
`plot speed` and `write speed` are applicable only in the shock-attached frame of reference.
* `write seconds`: show the simulation time every 0.01 units.
* `write video`: save `mp4` file with evolution. `ffmpeg` is required.
* `plot final solution`: save the graph with the solution at the final time.
* `write final solution`: save the `.npz` file with the solution at the final time.
The file contains array of nodes and array of solution values at these nodes.
* `plot speed`: save the graph with the shock speed evolution in time.
* `write speed`: save the `.npz` file with the shock speed evolution in time.
The file contains array of timestamps and array of shock speed values at these timestamps.

## Running a simulation
The simplest way to run your simulation is to edit `config.yaml` file in `src` directory
and then just run `python3 main.py` from there. If you want to work in another folder, you can specify
a path for `python` to find the solver via `export PYTHONPATH=/path/to/src` and than
run `python3 /path/to/main.py` having prepared the `config.yaml` inside your folder.

# NONUNIFORM DETONATION
Here you can find more details on the main problem of interest that is propagation of a
1D detonation wave in non-uniform media.


The governing equations are the reactive Euler equations in the shock-attached frame of reference [3,4]

```math
\begin{align}
& \frac{\partial\rho}{\partial t} + \frac{\partial}{\partial x}\left( \rho (u-D)\right) = 0 \\
& \frac{\partial}{\partial t} \left( \rho u\right) + \frac{\partial}{\partial x}\left( \rho u (u-D)+ p\right) = 0 \\
& \frac{\partial}{\partial t} \left(\rho \left(e + \frac 12 u^2\right)\right) + \frac{\partial}{\partial x} \left(\rho (u-D) \left(e + \frac 12 u^2 \right) + u p\right)= 0 \\
& \frac{\partial}{\partial t} \left( \rho \lambda \right)  + \frac{\partial}{\partial x}\left( \rho (u-D)\lambda \right) = K \rho (1-\lambda) \exp\left(-\frac{\rho E}{p}\right) \\
\end{align}
```

The Rankine-Hugoniot conditions are usual for these equations
```math
\begin{align}
 & \rho_s \left(D-u_s\right) = \rho_a \left(D-u_a\right) \\
 & p_s - p_a = \left(\rho_a \left(D - u_a\right)\right)^2 \left(\frac 1\rho_a -\frac 1\rho_s\right)\\
 & e_s - e_a = \frac 12\left(p_s + p_a\right) \left(\frac 1\rho_a -\frac 1\rho_s\right)\\
 & \lambda_s = \lambda_a
\end{align}
```
The shock-change equation for the shock velocity $D(t)$ is
```math
$$ \frac{dD}{dt} = - \left(\frac{d(\rho_s u_s)}{dD}\right)^{-1}\left.\left(
\frac{\partial(\rho u (u-D)+p)}{\partial x} + D\frac{d\left(\rho_s
u_s\right)}{d\xi}\right)\right|_{x=0} + \frac{c_f \rho u |u|}{2}$$
```

The periodic conditions ahead of the shock are modeled as follows
<!-- in two ways presented below. -->

<!-- RDE-like: -->

```math
\begin{align}
 \rho_a(\xi) & = \frac{p_a}{R T_a + R A\left(1 + \sin\left(2 \pi k \xi\right)\right)} \\
 u_a(\xi) & = 0\\
 p_a(\xi) & = p_a\\
 \lambda_a(\xi) & = A\left(1 + \sin\left(2 \pi k \xi\right)\right)
\end{align}
```

<!-- Periodic heat release: -->
<!-- ```math -->
<!-- $$ Q_a(\xi) = Q(1 + A \sin(2 \pi k \xi)) $$ -->
<!-- ``` -->

The numerical algorithm is TVD Runge-Kutta (3rd order) + WENO5M spatial fluxes interpolation from [[1]](#1).

#### References

<a id="1">[1]</a> [Henrick, A. K., Aslam, T. D. & Powers, J. M. Simulations of pulsating one-dimensional detonations with true fifth order accuracy. Journal of Computational Physics 213, 311–329 (2006).](https://www.sciencedirect.com/science/article/pii/S0021999105003827)

<a id="2">[2]</a> [Henrick, A.K., Aslam, T.D., & Powers, J.M. (2005) Mapped weighted essentially non-oscillatory schemes: Achieving optimal order near critical points. Journal of Computational Physics, 207, 542–567.](https://www.sciencedirect.com/science/article/pii/S0021999105000409)

<a id="3">[3]</a> [Kasimov, A. R. & Stewart, D. S. On the dynamics of self-sustained one-dimensional detonations: A numerical study in the shock-attached frame. Physics of Fluids 16, 3566–3578 (2004).](https://aip.scitation.org/doi/10.1063/1.1776531)

<a id="4">[4]</a> [R. Semenko, L. Faria, A. Kasimov, B. Ermolaev, Set-valued solutions for non-ideal detonation, Shock Waves, 26(2), 141–160, 2016](https://link.springer.com/article/10.1007/s00193-015-0610-3)

