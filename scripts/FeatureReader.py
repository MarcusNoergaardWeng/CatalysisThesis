import numpy as np
from collections import Counter
import itertools as it

class FeatureReader():
	'Parent feature reader class'
	def __init__(self, metals, n_atoms_site):
		self.metals = metals
		self.n_metals = len(metals)
		self.n_atoms_site = n_atoms_site
		self.ads_sites = list(it.combinations_with_replacement(metals, n_atoms_site))
		self.n_ads_sites = len(self.ads_sites)
	
	def get_features(self, slab, radius, zone_args, site_ids=None):
		'''Return the feature vector.
		Convenient function since its functionality is shared among the various sub-classes
		but for different zone parameters'''
		
		if site_ids is None:
		
			# Get adsorption site atom indices
			site_ids = slab.get_adsorption_site_ids(radius)
			
			assert len(site_ids) == self.n_atoms_site, f'The given adsorption site has {len(site_ids)} adsorbing atoms, but {self.n_atoms_site} was expected'
		
		else:
			
			# Convert atom indices to the central piece of the 3x3 slab for which the
			# slab object is defined
			site_ids = slab.original_to_3x3_index(site_ids)
			
		# Get the number of zones, apart from the adsorption site
		n_zones = len(zone_args)
		
		# Initiate feature list
		features = np.zeros(self.n_ads_sites + self.n_metals*n_zones, dtype=int)
		
		# Get symbols of site
		site_symbols = slab.symbols[site_ids]
		
		# Order the symbols according to the order given in metals
		site_symbols = tuple(sorted(site_symbols, key=self.metals.index))
		
		# Update features with the site
		features[list(self.ads_sites).index(site_symbols)] = 1
		
		# Get center of site
		center = slab.get_mean_location(site_ids)
		
		# Iterate through zone arguments
		for zone_idx, (start, stop, layer) in enumerate(zone_args):
			
			# Get atom indices in the current zone
			atom_ids = slab.get_closest_ids(center, start, stop, layer)
			
			# Get corresponding chemical symbols
			symbols = slab.symbols[atom_ids]
			
			# Count the occurences of each symbol
			count = Counter(symbols)
			
			# Append the count to the feature vector
			start_idx = self.n_ads_sites + zone_idx*self.n_metals
			stop_idx = self.n_ads_sites + (zone_idx+1)*self.n_metals
			features[start_idx : stop_idx] = [count[m] for m in self.metals]
		
		return features
		
		
	def write_to_csv(self):
		pass

class OntopStandard111(FeatureReader):
	'Class for reading on-top site features,'
	'including only the immediate neighboring atoms'
	def __init__(self, metals):
		super().__init__(metals, n_atoms_site=1)
		
	def get_features(self, slab, radius=2.6, site_ids=None):
		'Return features of the given Slab object'
		# Get features by calling the parent function with the appropriate zone parameters
		return super().get_features(slab, radius, zone_args=[(1,6,1), (0,2,2), (3,5,3)], site_ids=site_ids)
		
class FccStandard111(FeatureReader):
	'Class for reading fcc hollow site features'
	def __init__(self, metals):
		super().__init__(metals, n_atoms_site=3)
		
	def get_features(self, slab, radius=2.6, site_ids=None):
		'Return features of the given Slab object'
		# Get features by calling the parent function with the appropriate zone parameters
		return super().get_features(slab, radius, zone_args=[(3,5,1), (6,11,1), (0,2,2), (3,5,2)], site_ids=site_ids)
		
		
