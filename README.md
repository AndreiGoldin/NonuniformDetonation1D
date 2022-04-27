# NONUNIFORM DETONATION
This package solves the problem of propagation of a 1D detonation wave in non-uniform media

## STRUCTURE:
* `src/fp`:  Solver in functional paradigm with `numba`
* `src/oop`: Solver in OOP paradigm with `Cython` (see Langtangen&Linge 2017)
* `src/sbatch_scripts`: Submission and run scripts for SBATCH sheduler (@hilbert, @zhores)
* `src/pbs_scripts`:    Submission and run scripts for PBS sheduler (@arkuda)

## NOTES:
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

## TODO:
1. Simplify and refactor the code.
    * Implement pure functional version with `numba`
        * Callback functionality?
    * Use classes and `Cython` (see Langtangen&Linge 2017 and hil:~/Codes/fdm-book)
    * SOLID principles for OOP
        * Single responsibility: separate methods in classes
        * Open/closed: open to extension but closed for modification.
        * Liskov's substitution:  if S is a subtype of T, then objects of type T
          may be replaced by objects of type S, without breaking the program.
        * Interface segregation: separating interfaces into the smallest possible units
        * Dependency inversion: the details should depend on abstractions
2. Output with speed and fields
3. Test and check convergence. Write other tests. (see Fu et al. 2016)
    * Sod problem
    * Shu-Osher problem
    * Stable detonation speed (Henrick et al. 2006)
4. Add modules to handle the visual analysis (plots, animations,
   spectrograms, bifurcation diagrams, etc.)
5. Try to manage secondary shocks near the leading shock where one-sided differences are used
6. Arnold tongues for the periodic friction.
7. Quasiperiodic or noisy conditions without friction.
8. Closely analyse the problem of interaction detonation wave
   with one half of the sine wave
9. Increase performance: do not recalculate the stencils every time: decorators
   for cashing and logging.
