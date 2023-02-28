import numpy as np
from ase.build import fcc111, molecule
from ase.db import connect
from ase.constraints import FixAtoms
from ase.visualize import view
import json
from copy import deepcopy
import sys
sys.path.append('../..')
from shared_params import metals, lattice_parameters

# Specify size of slab
size = (3,3,5)
n_atoms = np.prod(size)
n_atoms_surface = np.prod(size[:2])

# Connect to output database
with connect('single_element_slabs.db') as db:

	# Iterate through slabs
	for metal in metals:
	
		# Generate chemical symbols
		symbols = [metal]*n_atoms

		# Make slab
		slab = fcc111('X', size=size, a=lattice_parameters[metal], vacuum=10.)
	
		# Fix all but the two top layers of atoms
		constraint = FixAtoms(indices=[atom.index for atom in slab if atom.tag > 2])
		slab.set_constraint(constraint)
	
		# Set chemical symbols
		slab.set_chemical_symbols(symbols)
		
		# Save slab atoms object to database
		db.write(slab, metal=metal)
