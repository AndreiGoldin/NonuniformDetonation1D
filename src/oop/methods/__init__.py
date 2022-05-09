# Provides Facade for methods
from .spatial import SpaceMethod
from .temporal import *

possible_integrators = {'ExplicitEuler': explicit_euler, 'TVDRK3': tvd_rk3, 'TVDRK5': tvd_rk5}
