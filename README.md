# DESCRIPTION
This package contains a Python solver for 1D hyperbolic equations and systems. Main purpose is to
implement the algorithm for the reactive Euler equations in the shock-attached frame of reference (SAFOR)
that is presented in [[6]](#6). However, in order to test different time and spatial discretization
methods, the advection, Burgers, and Euler equations are also included as well as the possibility
to work in the laboratory frame of reference. To enhance performance, [`numba` package](https://numba.pydata.org/) is used.
# INSTALLATION
To use the package, you need Python version >=3.6 and I recommend to use a separate
virtual environment for simulations, for example, via [`venv` package](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).
Please execute following commands in the terminal.
```
git clone https://github.com/AndreiGoldin/NonuniformDetonation1D.git
cd NonuniformDetonation1D
python3 -m venv solver-env
source solver-env/bin/activate
pip install -r requirements.txt
```
To test the installation run
```
python3 examples/simple_advection.py
```
If you want to have animations of the solution evolution, you need to install [`ffmpeg`](https://ffmpeg.org/download.html).
NB: When writing video is enabled, the solver works slower due to saving frames.
# USAGE
## Configuration
## Problems
## Methods
## Initial and boundary conditions
## Upstream conditions
# NONUNIFORM DETONATION
This package solves the problem of propagation of a 1D detonation wave in non-uniform media

![Problem.png](attachment:Problem.png)

The governing equations are the reactive Euler equations in the shock-attached frame of reference [4,5]

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
The shock-change equation for the shock velocity $`D(t)`$ is
```math
$$ \frac{dD}{dt} = - \left(\frac{d(\rho_s u_s)}{dD}\right)^{-1}\left.\left(
\frac{\partial(\rho u (u-D)+p)}{\partial x} + D\frac{d\left(\rho_s
u_s\right)}{d\xi}\right)\right|_{x=0} + \frac{c_f \rho u |u|}{2}$$
```

The periodic conditions ahead of the shock are modeled in two ways presented below.

RDE-like:

```math
\begin{align}
 \rho_a(\xi) & = \frac{p_a}{R T_a + R A\left(1 + \sin\left(2 \pi k \xi\right)\right)} \\
 u_a(\xi) & = 0\\
 p_a(\xi) & = p_a\\
 \lambda_a(\xi) & = A\left(1 + \sin\left(2 \pi k \xi\right)\right)
\end{align}
```

Periodic heat release:
```math
$$ Q_a(\xi) = Q(1 + A \sin(2 \pi k \xi)) $$
```

The numerical algorithm is TVD Runge-Kutta (3rd order) + WENO5M spatial fluxes interpolation from [6].

#### References

[1] Harten, A. & Osher, S. (1987) Uniformly High-Order Accurate Nonoscillatory Schemes. I. SIAM J. Numer. Anal., 24, 279–309.

[2] Jiang, G.-S. & Shu, C.-W. (1996) Efficient Implementation of Weighted ENO Schemes. Journal of Computational Physics, 126, 202–228.

[3] Henrick, A.K., Aslam, T.D., & Powers, J.M. (2005) Mapped weighted essentially non-oscillatory schemes: Achieving optimal order near critical points. Journal of Computational Physics, 207, 542–567.

[4] [Kasimov, A. R. & Stewart, D. S. On the dynamics of self-sustained one-dimensional detonations: A numerical study in the shock-attached frame. Physics of Fluids 16, 3566–3578 (2004).](https://www.researchgate.net/publication/32955598_On_the_dynamics_of_self-sustained_one-dimensional_detonations_A_numerical_study_in_the_shock-attached_frame)

[5] [R. Semenko, L. Faria, A. Kasimov, B. Ermolaev, Set-valued solutions for non-ideal detonation, Shock Waves, 26(2), 141–160, 2016](https://link.springer.com/article/10.1007/s00193-015-0610-3)

<a id="6">[6]</a> [Henrick, A. K., Aslam, T. D. & Powers, J. M. Simulations of pulsating one-dimensional detonations with true fifth order accuracy. Journal of Computational Physics 213, 311–329 (2006).](https://www.sciencedirect.com/science/article/pii/S0021999105003827)

## STRUCTURE:
* `src/fp`:  Solver in functional paradigm with `numba`
* `src/oop`: Solver in OOP paradigm with `Cython` (see Langtangen&Linge 2017)
* `src/sbatch_scripts`: Submission and run scripts for SBATCH sheduler (@hilbert, @zhores)
* `src/pbs_scripts`:    Submission and run scripts for PBS sheduler (@arkuda)

## NOTES ON PROGRAMMING:
Martin Fowler in his book Refactoring: Improving the Design of Existing Code
defines refactoring as “the process of changing a software system in such a way
that does not alter the external behavior of the code yet improves its internal
structure.”

Don't Repeat Yourself
Keep It Simple Stupid
SOLID principles
    * Single responsibility: separate methods in classes
    * Open/closed: open to extension but closed for modification.
    * Liskov's substitution:  if S is a subtype of T, then objects of type T
      may be replaced by objects of type S, without breaking the program.
    * Interface segregation: separating interfaces into the smallest possible units
    * Dependency inversion: the details should depend on abstractions
A good rule of thumb is that any given function shouldn't take longer than a
few minutes to comprehend.  As a general rule, if you need to add comments,
they should explain "why" you did something rather than "what" is happening.

Having maintainable software is not about anticipating future requirements (do
not do futurology!). It is about writing software that **only addresses current
requirements in such a way that it will be possible (and easy) to change later on.**

A pragmatic approach is to first make a quick function-based code, but refactor
that code to **a more reusable and extensible class version with test functions**
when you experience that the code frequently undergo changes.

We’re also going to talk about the change in thinking that they enable which is
**key to functional thinking: parameterizing code by behavior.** It’s this thinking
in terms of functions and parameterizing by behavior rather than state which is
key to differentiating func‐ tional programming from object-oriented
programming.

**First-class functions**: idea that we can pass behavior around and treat it like another value.
**Higher-order functions** are just functions, methods, that return other functions
or take functions as a parameter.

SO for FP: Higher-order functions
L for FP: Inheritance of behavior isn’t a key trait. No inheritance, no problem!
I for FP: Irrelevant due to structural subtyping
D for FP: Higher-order functions provide an inversion of control

### Design Patterns
Patterns codify what people consider to be a best-practice approach to a given problem.

*Facade*
Provide a unified interface to a set of interfaces in a subsystem. Facade
defines a higher-level interface that makes the subsystem easier to use.
Collaborations
* Clients communicate with the subsystem by sending requests to Facade, which
forwards them to the appropriate subsystem object(s). Although the subsystem
objects perform the actual work, the facade may have to do work of its own to
translate its interface to subsystem interfaces.
* Clients that use the facade don't have to access its subsystem objects directly.
Python: No need for a class, just organize code in high-level packages and write
Facade API in `__init__.py`.
[Video](https://www.youtube.com/watch?v=G5OeYHCJuv0)

*Bridge*
Decouple an abstraction from its implementation so that the two can vary independently.
Inheritance binds an implementation to the abstraction permanently, which
makes it difficult to modify, extend, and reuse abstractions and
implementations independently.
Collaborations: Abstraction forwards client requests to its Implementor object.

*Strategy*
Define a family of algorithms, encapsulate each one, and make them interchange-
able. Strategy lets the algorithm vary independently from clients that use it.
[Python example](https://auth0.com/blog/strategy-design-pattern-in-python/)
Collaborations
* Strategy and Context interact to implement the chosen algorithm. A context
may pass all data required by the algorithm to the strategy when the algorithm
is called. Alternatively, the context can pass itself as an argument to
Strategy operations. That lets the strategy call back on the context as
required.
* A context forwards requests from its clients to its strategy.
Clients usually create and pass a ConcreteStrategy object to the context;
thereafter, clients interact with the context exclusively. There is often a
family of ConcreteStrategy classes for a client to choose from.

*Mediator* (maybe, for complex communications in the coupled PDE system in SAFOR)
Define an object that encapsulates how a set of objects interact. Mediator
promotes loose coupling by keeping objects from referring to each other
explicitly, and it lets you vary their interaction independently.
Collaborations: Colleagues send and receive requests from a Mediator object.
The mediator implements the cooperative behavior by routing requests between
the appropriate colleague(s).
Issues: There's **no need to define an abstract Mediator class** when colleagues
work with only one mediator. Can be implemented as an Observer.

*Observer* (maybe, to provide a callback, e.g. write ooutput at particular time)
Define a one-to-many dependency between objects so that when one object changes
state, all its dependents are notified and updated automatically.
Observer provide a callback for notification of events/changes to data.
A simpler approach is to just implement the callback since the Observer pattern
implies a synchronization between a subject and its observers that is not necessary
for just writing output.

*Abstract Factory*
Provide an interface for creating families of related or dependent objects
without specifying their concrete classes.

*Factory*
Define an interface for creating an object, but let subclasses decide which
class to instantiate. Factory Method lets a class defer instantiation to
subclasses.


A *command object* is an object that encapsulates all the information required to
call another method later. The *command pattern* is a way of using this object in
order to write generic code that sequences and executes methods based on
runtime decisions. There are four classes that take part in the command pattern:
*Receiver*
Performs the actual work
*Command*
Encapsulates all the information required to call the receiver
*Invoker*
Controls the sequencing and execution of one or more com‐ mands
*Client*
Creates concrete command instances
The command pattern is all about **first-class functions.**

The *strategy pattern* is a way of changing the algorithmic behavior of software
based upon a runtime decision.
It’s a design pattern that’s mimicking first-class functions.

The future is hybrid: pick the best features and ideas from both functional and
object-oriented approaches in order to solve the problem at hand.

In the [*Factory Method*](https://realpython.com/factory-method-python/), the
**creator** component decides which concrete
implementation to use. There are also **client** and **product** components.
The former is application code that depends on an interface to complete its
task. The latter is the defined interface that has some concrete
implementations being chosen by the **creator**. A client
depends on a concrete implementation of an interface. It requests the
implementation from a creator component using some sort of identifier.
The creator returns the concrete implementation according to the value of the
parameter to the client, and the client uses the provided object to complete
its task.

Factory Method should be used in every situation where an application (client)
depends on an interface (product) to perform a task and **there are multiple
concrete implementations of that interface**. You need to provide a parameter
that can identify the concrete implementation and use it in the creator to
decide the concrete implementation.

By implementing Factory Method using an **Object Factory** and providing a
**registration interface**, you are able to support new formats without changing
any of the existing application code. This minimizes the risk of breaking
existing features or introducing subtle bugs.

The pattern removes complex logical code that is hard to maintain, and replaces
it with a design that is reusable and extensible. The pattern avoids modifying
existing code to support new requirements.

### Scientific computing patterns
From [this article](https://ccc.inaoep.mx/~grodrig/Descargas/429-155.pdf)
1. The integrator–class structure must be reusable without modification.
2. The integrator–class must not contain details of the integrated model.
3. The differential equations are located outside the integrator–class.
4. The integrator–class must be extensible. New integrations methods can be
   incorporated within the integrator–class structure with minimal coding.
5. The numerical methods are reusable for a large set of ordinary differential equations.
6. The selected numerical method is instantiated at compilation time.

Scientific computing patterns capture the fundamental characteristics of the
problem to be solved. They are focused on “what” and not in “how ”.

### Cython
" Type declarations can therefore be used for two purposes: for moving code
sections from dynamic Python semantics into static-and-fast C semantics, but
also for directly manipulating types defined in external libraries.
Cython thus merges the two worlds into a very broadly applicable programming language. "

In contrast to numba, Cython can compile arbitrary Python code, and can even directly
call C. The ability to “cythonize” an entire module written using advanced
Python features and then only tweak the bottlenecks for speed can be really
nice.

### Numba
The best part of Numba is that it neither needs separate compilation step nor
needs major code modification.
There is `numba.typed.Dict`
[But most importantly, Numba breaks down when we add a minimal higher-level
construction.](https://www.matecdev.com/posts/julia-python-numba-cython.html)

### Tests
Writing tests before, such as the case of Test Driven Development, or while you
write your logic actually shapes the way code is written. It leads to modular,
extensible code that is easy to test, understand, and maintain.
Code coverage: nose, code complexity: pygenie, pylint.
Unittest, doctest, functional, regression tests, automatic

## TODO:
1. Simplify and refactor the code.
    * Write abstract flexible and extensible solver and then wrap critical methods in `numba`
        * Test performance of such wrappers show that the strategy leads to speed up.
    * OOP approach with `numba`: @staticmethod and/or wrappers. See
        [this post](https://stackoverflow.com/questions/41769100/how-do-i-use-numba-on-a-member-function-of-a-class)
    * Use classes and `Cython` (see Langtangen&Linge 2017 and hil:~/Codes/fdm-book)
        * `Cython` needs a lot of code when `numpy` is used
    * Implement pure functional version with `numba` (maybe, give up `numba` and use only `Cython`?)
        * Callback functionality?
2. Output with speed and fields
3. Test and check convergence. Write other tests. (see Fu et al. 2016)
    * Sod problem
    * Shu-Osher problem
    * Stable detonation speed (Henrick et al. 2006)
4. Add modules to handle the visual analysis (plots, animations, x-t diagrams,
   spectrograms, bifurcation diagrams, etc.)
5. Try to manage secondary shocks near the leading shock where one-sided differences are used
6. Arnold tongues for the periodic friction.
7. Quasiperiodic or noisy conditions without friction.
8. Closely analyse the problem of interaction detonation wave
   with one half of the sine wave
9. Increase performance: do not recalculate the stencils every time: decorators
   for cashing and logging.
