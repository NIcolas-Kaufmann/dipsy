from importlib import metadata as _md
from . import data
from . import utils
from . import distributions 
from . import cgs_constants
from .tracks import get_stellar_properties
from dustpylib.radtrans.slab.slab import Opacity
try:
    from dipsy._fortran_module import fmodule as fortran
except ImportError:
    print('fortran module not available')

__version__ = _md.version('dipsy')

from .dipsy_functions import \
    get_all_observables, \
    get_observables, \
    get_powerlaw_dust_distribution, \
    read_dustpy_data, \
    read_rosotti_data, \
    read_tazzari_data, \
    read_andrews_data, \
    read_bito_data, \
    read_feng_data

__all__ = [
    'data',
    'utils',
    'cgs_constants',
    'Opacity',
    'get_all_observables',
    'get_observables',
    'get_powerlaw_dust_distribution',
    'read_dustpy_data',
    'read_rosotti_data',
    'read_tazzari_data',
    'read_andrews_data',
    'read_bito_data',
    'get_stellar_properties'
]
