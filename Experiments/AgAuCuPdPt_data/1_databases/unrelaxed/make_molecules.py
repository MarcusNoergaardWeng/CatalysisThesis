from ase.build import molecule
from ase.db import connect
from ase.visualize import view

# Make molecules
labels = [
	'CO',
	'CO2',
	'HCOOH',
	'CH3COOH',
	'H2',
	'H2O',
	'H2CO',
	'CH3OH',
	'CH3CH2OH',
	'OCHCHO', # glyoxal
	'HOOCCOOH' # oxalic acid
	]

def make(name):
	if name == 'HOOCCOOH':
		# Make oxalic acid
		HCOOH_1 = molecule('HCOOH')[:-1]
		HCOOH_1.positions -= HCOOH_1.positions[1]

		HCOOH_2 = molecule('HCOOH')[:-1]
		HCOOH_2.positions -= HCOOH_2.positions[1] + [0.0, 1.3, 0.0]
		HCOOH_2.rotate(180, (0,0,1))

		return HCOOH_1 + HCOOH_2
	raise KeyError(f'{name} is not defined')

# Connect to input and output databases
with connect('molecules.db') as db:
	
	# Iterate through molecules
	for label in labels:
		
		try:
			# Make atoms object
			atoms = molecule(label)
		except KeyError:
			atoms = make(label)
		
		# Make unit cell around molecule
		atoms.set_cell((20., 20., 20.))
		
		# Set periodic boundary conditions
		atoms.set_pbc((True, True, True))
		
		# Center molecule in the unit cell
		atoms.center()
		
		# Save atoms object to database
		db.write(atoms)
