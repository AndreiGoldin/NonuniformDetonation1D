# Provides Facade for problems
from .equations import Equations
from .initial_conditions import *
from .boundary_conditions import *
from .upstream_conditions import *

possible_bc = {'Zero': boundary_lfor_zero}
possible_ic = {'ZND': initial_ZND, 'Sine':initial_sine}
possible_uc = {'RDE': upstream_RDE}
