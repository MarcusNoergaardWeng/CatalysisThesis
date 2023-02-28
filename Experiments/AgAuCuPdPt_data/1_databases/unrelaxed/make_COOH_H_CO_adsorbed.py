import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.db import connect
from ase.constraints import FixAtoms
from copy import deepcopy
import sys
sys.path.append('../..')
from shared_params import metals

# Specify size of slab
size = (3,3,5)
n_atoms = np.prod(size)
n_atoms_surface = np.prod(size[:2])

# Make COOH adsorbate
HCOOH = molecule('HCOOH')
COOH_orig = HCOOH[:-1]
COOH_orig.rotate(-90, (1,0,0))

# Make H adsorbate
H_orig = Atoms('H')

# Put carbon atom at (0,0,0)
C_pos = [atom.position for atom in COOH_orig if atom.symbol == 'C']
COOH_orig.positions -= C_pos[0]

# Rotate adsorbate around y-axis
COOH_orig.rotate(15, (0,1,0))

# Set height above surface to put adsorbate at
height = 1.9 # A
height_H = 0.9 # A

def add_COOH_H(slab, site_idx=0):
	
	# Get position of on-top site
	site_position = slab[n_atoms - n_atoms_surface + site_idx].position

	# Set position of the adsorbate
	COOH = deepcopy(COOH_orig)
	COOH.positions += site_position + [0.8, 0., 0.] # Displace a little extra along the x-axis to make the oxygens be approximately equidistant to a surface atom
	COOH.positions[:, 2] += height
	
	# Set *H on surface
	H = deepcopy(H_orig)
	H.positions += site_position + [1.41, 0.82, 0.]
	H.positions[:, 2] += height_H
	
	# Add the adsorbate to the on-top site
	return slab + COOH + H

# Connect to input and output databases
with connect('single_element_COOH_H_CO_adsorbed.db') as db_ads,\
	 connect('single_element_slabs.db') as db_slab:

	# Iterate through slabs
	for metal in metals:
		
		# Get slab atoms object
		slab = db_slab.get_atoms(metal=metal)
		
		# Get atoms with COOH added
		atoms = add_COOH_H(slab)
		
		# Save atoms object to database
		db_ads.write(atoms, metal=metal)
