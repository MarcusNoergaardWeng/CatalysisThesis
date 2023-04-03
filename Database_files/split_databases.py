from ase.db import connect
from collections import Counter

#with connect('monodentate_out.db') as db_in,\
#	 connect('COOH_C_adsorbed_out.db') as db_ads,\
#	 connect('slabs_out.db') as db_slab:
#	
#	for row in db_in.select():
#		
#		atoms = db_in.get_atoms(row.id)
#		n_each = Counter(atoms.get_chemical_symbols())
#		
#		if n_each['C'] == n_each['O'] == n_each['H'] == 0:
#			db_slab.write(atoms, idx=row.idx, slab_idx=row.slab_idx)
#		
#		else:
#			db_ads.write(atoms, idx=row.idx, slab_idx=row.slab_idx, site_idx=row.site_idx)

with connect('monodentate_pure_out.db') as db_in,\
	 connect('single_element_COOH_C_adsorbed_out.db') as db_ads,\
	 connect('single_element_slabs_out.db') as db_slab:
	
	for row in db_in.select():
		
		atoms = db_in.get_atoms(row.id)
		n_each = Counter(atoms.get_chemical_symbols())
		
		if n_each['C'] == n_each['O'] == n_each['H'] == 0:
			db_slab.write(atoms, metal=row.metal)
		
		else:
			db_ads.write(atoms, metal=row.metal)
