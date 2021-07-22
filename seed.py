"""
Seed-related functions: init, display, save, load,...
"""
import random
import os
import numpy as np
import torch

############################### SEED INITIALIZER ###################################

def init_seeds(_seed, _cuda=True):
	"""Initialize the ``random.seed()``, ``np.random.seed()``, ``torch.manual_seed()``,
	and ``torch.cuda.manual_seed()`` if **args** ``.cuda`` is True with the value in **args** ``.seed``.

	Args:
		* **args** (:xref:namespace:`Namespace <>` or ``fox_nn.base_fct.empty_args()``): An arg namespace containing the field 'seed' whose value (:xref:int:`int <>`) is used to initialize the generators
	"""
	random.seed(_seed)
	np.random.seed(_seed)
	torch.manual_seed(_seed)
	if _cuda:
		torch.cuda.manual_seed(_seed)
