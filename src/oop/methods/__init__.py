# Provides Facade for methods
from .spatial import SpaceMethod
from .temporal import *

possible_integrators = {'ExplicitEuler': explicit_euler,
                        'EULER_SAFOR': euler_safor,
                        'TVDRK3': tvd_rk3,
                        'TVDRK3_SAFOR':tvd_rk3_safor,
                        'TVDRK5': tvd_rk5}
