import numpy as np
from ase.build import molecule
from ase.db import connect
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

# Put carbon atom at (0,0,0)
C_pos = [atom.position for atom in COOH_orig if atom.symbol == 'C'][0]
COOH_orig.positions -= C_pos

# Set height above surface to put adsorbate at
height = 1.9 # A

# Set number of slabs to make
n_slabs = 56

def add_COOH_C_adsorbed(slab, site_idx=0):
	'Add COOH to slab at the specified site_idx'

	# Get position of on-top site
	site_position = slab[n_atoms - n_atoms_surface + site_idx].position

	# Set position of the adsorbate
	COOH = deepcopy(COOH_orig)
	COOH.positions += site_position
	COOH.positions[:, 2] += height

	# Add the adsorbate to the on-top site
	return slab + COOH

# Connect to output database
with connect('COOH_C_adsorbed.db') as db_ads,\
	 connect('slabs.db') as db_slab:

	# Iterate through slabs
	for slab_idx in range(n_slabs):
	
		# Get slab from database
		slab = db_slab.get_atoms(slab_idx=slab_idx)

		# Iterate through surface on-top sites
		for site_idx in range(n_atoms_surface):
			
			# Get atoms with COOH adsorbed on carbon atom
			atoms = add_COOH_C_adsorbed(slab, site_idx)
		
			# Save atoms object to database
			db_ads.write(atoms, slab_idx=slab_idx, site_idx=site_idx)

# Add COOH to single element slabs
with connect('single_element_COOH_C_adsorbed.db') as db_ads,\
	 connect('single_element_slabs.db') as db_slab:

	# Iterate through metals
	for metal in metals:
		
		# Get slab atoms object
		slab = db_slab.get_atoms(metal=metal)
		
		# Get atoms with COOH added
		atoms = add_COOH_C_adsorbed(slab)
		
		# Save atoms object to database
		db_ads.write(atoms, metal=metal)
