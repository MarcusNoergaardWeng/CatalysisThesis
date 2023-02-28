import numpy as np
from ase.build import fcc111, molecule
from ase.db import connect
from ase.constraints import FixAtoms
from ase.visualize import view
import json
from copy import deepcopy
import sys
sys.path.append('../..')
from shared_params import metals

# Specify size of slab
size = (3,3,5)
n_atoms = np.prod(size)
n_atoms_surface = np.prod(size[:2])

# Make COOH adsorbate
HCOOH_orig = molecule('HCOOH')
HCOOH_orig.rotate(90, (0,1,0))

# Put second oxygen atom at (0,0,0)
O_pos = [atom.position for atom in HCOOH_orig if atom.symbol == 'O']
HCOOH_orig.positions -= O_pos[1]

# Set height above surface to put adsorbate at
height = 1.9 # A

# Set number of slabs to make
n_slabs = 3

def add_HCOOH_O_adsorbed(slab, site_idx=0):
	
	# Get position of on-top site
	site_position = slab[n_atoms - n_atoms_surface + site_idx].position

	# Set position of the adsorbate
	HCOOH = deepcopy(HCOOH_orig)
	HCOOH.positions += site_position
	HCOOH.positions[:, 2] += height

	# Add the adsorbate to the on-top site
	return slab + HCOOH

# Connect to input and output databases
with connect('HCOOH_O_adsorbed.db') as db_ads,\
	 connect('slabs.db') as db_slab:

	# Iterate through slabs
	for slab_idx in range(n_slabs):
		
		# Get slab atoms object
		slab = db_slab.get_atoms(slab_idx=slab_idx)
		
		# Iterate through surface on-top sites
		for site_idx in range(n_atoms_surface):
			
			# Get atoms with COOH added
			atoms = add_HCOOH_O_adsorbed(slab, site_idx)
			
			# Save atoms object to database
			db_ads.write(atoms, slab_idx=slab_idx, site_idx=site_idx)

# Connect to input and output databases
with connect('single_element_HCOOH_O_adsorbed.db') as db,\
	 connect('single_element_slabs.db') as db_slab:

	# Iterate through slabs
	for metal in metals:
		
		# Get slab atoms object
		slab = db_slab.get_atoms(metal=metal, C=0, O=0, H=0)
		
		# Get atoms with COOH added
		atoms = add_HCOOH_O_adsorbed(slab)
		
		# Save atoms object to database
		db.write(atoms, metal=metal)