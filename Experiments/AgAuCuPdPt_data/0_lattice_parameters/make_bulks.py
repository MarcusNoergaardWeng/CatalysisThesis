from ase.build import bulk
from ase.db import connect
import numpy as np
from copy import deepcopy

db = connect('bulks.db')

lats_init = dict(Be = 3.233,
				 B  = 2.815,
				 Mg = 4.538,
				 Al = 4.050,
				 Si = 3.894,
				 Sc = 4.680,
				 Ti = 4.173,
				 V  = 3.857,
				 Cr = 3.648,
				 Mn = 3.534,
				 Fe = 3.706,
				 Co = 3.563,
				 Ni = 3.565,
				 Cu = 3.697,
				 Zn = 3.987,
				 Ga = 4.282,
				 Ge = 4.287,
				 Y  = 5.128,
				 Zr = 4.568,
				 Nb = 4.237,
				 Mo = 4.020,
				 Ru = 3.835,
				 Rh = 3.874,
				 Pd = 3.992,
				 Ag = 4.223,
				 Cd = 4.598,
				 In = 4.853,
				 Sn = 4.860,
				 Hf = 4.479,
				 Ta = 4.249,
				 W  = 4.054,
				 Re = 3.889,
				 Os = 3.833,
				 Ir = 3.892,
				 Pt = 4.004,
				 Au = 4.229,
				 Hg = 7.565,
				 Tl = 5.090,
				 Pb = 5.094,
				 Bi = 5.072)

magmoms_init = dict(Fe=2.8,
					Co=1.8,
					Ni=0.7)

metals = sorted(lats_init.keys())

for metal in metals:

	atoms = bulk(name=metal, crystalstructure='fcc', a=lats_init[metal])
	
	if metal in magmoms_init:
		magmoms = [magmoms_init[metal]]
		atoms.set_initial_magnetic_moments(magmoms)

	cell = deepcopy(atoms.cell)
	
	if metal == 'Fe':
		eps1 = 0.99
		eps2 = 1.01
	else:
		eps1 = 0.95
		eps2 = 1.05
	
	# Iterate through five unit cell sizes	
	for lat_idx, eps in enumerate(np.linspace(eps1, eps2, 5)):
		atoms.set_cell(cell*eps, scale_atoms=True)
		db.write(atoms, metal=metal, lat_idx=lat_idx)
