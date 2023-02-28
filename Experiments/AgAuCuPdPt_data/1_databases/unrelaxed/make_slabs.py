import numpy as np
from ase.build import fcc111
from ase.db import connect
from ase.constraints import FixAtoms
import sys
sys.path.append('../..')
from shared_params import metals, lattice_parameters

# Set random seed for reproducibility
np.random.seed(49226)

# Specify size of slab
size = (3,3,5)
n_atoms = np.prod(size)
n_atoms_surface = np.prod(size[:2])

# Set number of slabs to make
n_slabs = 56

# Connect to output database
with connect('slabs.db') as db:

	# Iterate through slabs
	for slab_idx in range(n_slabs):
	
		# Generate chemical symbols
		symbols = np.random.choice(metals, size=n_atoms)
	
		# Set the lattice parameter of the slab to the average of
		# the surface atoms' lattice constants
		surface_symbols = symbols[-n_atoms_surface:]
		lattice_param = sum(lattice_parameters[metal] for metal in surface_symbols) / n_atoms_surface
	
		# Make slab
		slab = fcc111('X', size=size, a=lattice_param, vacuum=10.)
	
		# Fix all but the two top layers of atoms
		constraint = FixAtoms(indices=[atom.index for atom in slab if atom.tag > 2])
		slab.set_constraint(constraint)
	
		# Set chemical symbols
		slab.set_chemical_symbols(symbols)
		
		# Save slab atoms object to database
		db.write(slab, slab_idx=slab_idx)
