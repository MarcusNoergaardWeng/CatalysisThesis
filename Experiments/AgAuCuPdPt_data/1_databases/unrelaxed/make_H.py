import numpy as np
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

# Make H adsorbate
H_orig = molecule('H')

# Put H atom at (0,0,0)
H_pos = H_orig.positions
H_orig.positions -= H_pos[0]

# Set height above surface to put adsorbate at
height = 1.0 # A

# Set number of slabs to make
n_slabs = 56

def add_H(slab, site_idx=0):
	
	# Get position of on-top site
	site_position = slab[n_atoms - n_atoms_surface + site_idx].position
	
	# Get interatomic distance between atom index 0 and 1
	dist = (np.sum((slab[1].position - slab[0].position)**2))**0.5
	
	# Set position of the adsorbate
	H = deepcopy(H_orig)
	H.positions += site_position + [dist/2, dist/(2*3**0.5), 0.]
	H.positions[:, 2] += height

	# Add the adsorbate to the on-top site
	return slab + H

# Connect to input and output databases
with connect('H.db') as db_ads,\
	 connect('slabs.db') as db_slab:

	# Iterate through slabs
	for slab_idx in range(n_slabs):
		
		# Get slab atoms object
		slab = db_slab.get_atoms(slab_idx=slab_idx)
		
		# Iterate through surface on-top sites
		for site_idx in range(n_atoms_surface):
			
			# Get atoms with H added
			atoms = add_H(slab, site_idx)
			
			# Save atoms object to database
			db_ads.write(atoms, slab_idx=slab_idx, site_idx=site_idx)

# Connect to input and output databases
with connect('single_element_H.db') as db_ads,\
	 connect('single_element_slabs.db') as db_slab:

	# Iterate through slabs
	for metal in metals:
		
		# Get slab atoms object
		slab = db_slab.get_atoms(metal=metal)
		
		# Get atoms with COOH added
		atoms = add_H(slab)
		
		# Save atoms object to database
		db_ads.write(atoms, metal=metal)
