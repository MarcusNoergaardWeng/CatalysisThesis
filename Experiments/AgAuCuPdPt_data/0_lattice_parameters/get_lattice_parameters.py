from ase.eos import EquationOfState
from ase.db import connect
import numpy as np
import json
import matplotlib.pyplot as plt

# Initiate dictionary of lattice parameters
lats = {}

db = connect('bulks_out.db')

volumes = {}
energies = {}

for row in db.select('energy'):
	
	if row.metal not in volumes:
		volumes[row.metal] = []
		energies[row.metal] = []
	
	volumes[row.metal].append(row.volume)
	energies[row.metal].append(row.energy)

for metal in volumes:
	
	# Get the lattice parameter with the minimum energy
	eos = EquationOfState(volumes[metal], energies[metal])
	
	try:
		V_min, E_min, _ = eos.fit()
		lat_min = 2*(V_min/2)**(1/3)
		lats[metal] = round(lat_min, 4)
		
		plt.subplots()
		eos.plot(f'pngs/{metal:s}.png')
		plt.close()
		
	except ValueError:
		print(f'[INFO] No minimum for {metal:s}')
		fig, ax = plt.subplots()
		ax.scatter(volumes[metal], energies[metal])
		fig.savefig(f'pngs/{metal:s}.png')
		plt.close()
		continue
	
# Save lattice parameters to json file
with open('lattice_parameters.json', 'w') as f:
	json.dump(lats, f, sort_keys=True, indent=2)
