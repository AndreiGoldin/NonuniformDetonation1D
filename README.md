This package solves the problem of propagation of a 1D detonation wave in non-uniform media

TODO
1. Simplify and refactor the code.
2. Test and check convergence. Write other tests.
3. Quasiperiodic or noisy conditions without friction.
4. Arnold tongues for the periodic friction.
5. Increase performance: do not recalculate the stencils every time: decorators
   for cashing and logging.
6. Try to manage secondary shocks near the leading shock where one-sided differences are used
7. Closely analyse the problem of interaction detonation wave
   with one half of the sine wave
