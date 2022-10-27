# Provides Facade for problems
from .equations import Equations
from .initial_conditions import *
from .boundary_conditions import *
from .upstream_conditions import *
from .exact_solutions import *

possible_bc = {
        'Zero gradient_SAFOR': boundary_safor,
        'Zero': boundary_lfor_zero,
        'Zero gradient': boundary_lfor_zero_grad,
        'PeriodicRight': boundary_periodic_right_going,
        'PeriodicLeft': boundary_periodic_left_going,
        'Periodic': boundary_none,
        'None':boundary_none
        }
possible_ic = {
        'ZND': initial_znd_lfor,
        'ZND_LFOR': initial_znd_lfor,
        'ZND_LFOR_halfwave': initial_znd_lfor_halfwave,
        'Sine':initial_sine,
        'Henrick2005':initial_henrick2005,
        'Shu-Osher':initial_shu_osher
        }
possible_uc = {
        'RDE': upstream_rde,
        'Uniform': upstream_uniform
        }
possible_es = {
        'Henrick2005': exact_henrick
        }
