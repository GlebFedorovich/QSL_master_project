import numpy as np
import scipy
import matplotlib.pyplot as plt
np.set_printoptions(precision=5, suppress=True, linewidth=100)
plt.rcParams['figure.dpi'] = 150
from scipy.io import savemat
from tenpy.tools.params import get_parameter
from matplotlib import colors
import matplotlib.pyplot as plt
import sys
from tenpy.models.mixed_xk import MixedXKLattice
from tenpy.models.lattice import Lattice
from tenpy.networks.terms import TermList

import numpy as np
import itertools as it
import warnings

import tenpy
import tenpy.linalg.np_conserved as npc
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
tenpy.tools.misc.setup_logging(to_stdout="INFO")
import pickle
from tenpy.tools.misc import to_array, inverse_permutation, to_iterable

from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
from tenpy.algorithms import dmrg
from tenpy.networks.site import SpinSite, SpinHalfSite, SpinHalfFermionSite, FermionSite
from tenpy.models.lattice import Triangular, Square
from tenpy.models.model import CouplingModel, NearestNeighborModel, MPOModel, CouplingMPOModel
