import numpy as np
from ase.build import molecule
from ase.db import connect
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
HCOOH = molecule('HCOOH')
COOH_orig = HCOOH[:-1]
COOH_orig.rotate(-90, (1,0,0))

# Put first oxygen atom at (0,0,0)
O_pos = [atom.position for atom in COOH_orig if atom.symbol == 'O']
COOH_orig.positions -= O_pos[0]

# Rotate adsorbate to make the oxygens point down
a = np.arctan(O_pos[1][2] / O_pos[1][0]) + np.pi
Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
COOH_orig.positions = (Ry @ COOH_orig.positions.T).T

# Rotate hydrogen to point up
H_idx = [atom.index for atom in COOH_orig if atom.symbol == 'H'][0]

# Rotate hydrogen location around oxygen
a = -140 * np.pi / 180
Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
COOH_orig.positions[H_idx] = (Ry @ COOH_orig.positions[H_idx].T).T

# Rotate adsorbate around z-axis
COOH_orig.rotate(180, (0,0,1))

# Set height above surface to put adsorbate at
height = 1.9 # A

# Set number of slabs to make
n_slabs = 56

def add_COOH_OO_adsorbed(slab, site_idx=0):
	'Add COOH to slab with adsorption at the two oxygens'
	
	# Get position of on-top site
	site_position = slab[n_atoms - n_atoms_surface + site_idx].position

	# Set position of the adsorbate
	COOH = deepcopy(COOH_orig)
	COOH.positions += site_position + [0.2, 0., 0.] # Displace a little extra along the x-axis to make the oxygens be approximately equidistant to a surface atom
	COOH.positions[:, 2] += height

	# Add the adsorbate to the on-top site
	return slab + COOH

# Connect to input and output databases
with connect('COOH_H_on_O_OO_adsorbed.db') as db_ads,\
	 connect('slabs.db') as db_slab:

	# Iterate through slabs
	for slab_idx in range(n_slabs):
		
		# Get slab atoms object
		slab = db_slab.get_atoms(slab_idx=slab_idx)
		
		# Iterate through surface on-top sites
		for site_idx in range(n_atoms_surface):
			
			# Get atoms with COOH added
			atoms = add_COOH_OO_adsorbed(slab, site_idx)
			
			# Save atoms object to database
			db_ads.write(atoms, slab_idx=slab_idx, site_idx=site_idx)

# Connect to input and output databases
with connect('single_element_COOH_H_on_O_OO_adsorbed.db') as db_ads,\
	 connect('single_element_slabs.db') as db_slab:

	# Iterate through slabs
	for metal in metals:
		
		# Get slab atoms object
		slab = db_slab.get_atoms(metal=metal)
		
		# Get atoms with COOH added
		atoms = add_COOH_OO_adsorbed(slab)
		
		# Save atoms object to database
		db_ads.write(atoms, metal=metal)
