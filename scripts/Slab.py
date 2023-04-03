from ase import Atoms
from ase.visualize import view
import re
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def inside_triangle(point, vertices):
	'Return True if the point is in the triangle interior spanned by the vertices'
	# Find a1 and a2 that solves the two equations
	# v = v0 + a1*v1 + a2*v2
	# where v is the point vector, and v0, v1, and v2 are the vertex vectors, the latter
	# two starting at v0
	vertices = deepcopy(vertices) # Taking a copy seems necessary to avoid this function from
								  # mutating the original ´vertices´ variable that was passed
	vertices = np.asarray(vertices)
	vertices[1] -= vertices[0]
	vertices[2] -= vertices[0]
	det1 = np.cross(point, vertices[1])
	det2 = np.cross(point, vertices[2])
	det01 = np.cross(vertices[0], vertices[1])
	det02 = np.cross(vertices[0], vertices[2])
	det12 = np.cross(vertices[1], vertices[2])
	a1 = (det2 - det02) / det12
	a2 = -(det1 - det01) / det12
	return 0 < a1 < 1 and 0 < a2 < 1

def expand_triangle(vertices, expansion):
	'Return vertices whose distances to their center has been extended by the'
	'specified expansion factor'
	# Get center of hollow site
	center = np.mean(vertices, axis=0)
	
	# Get vectors from the center to the corners
	diffs = vertices - center
	
	# Expand vertices from the center
	return center + diffs*expansion

class Slab(Atoms):
	'Class with handy extra functions for the ase.Atoms class'
	'for surfaces with adsorbates'
	
	def __init__(self, atoms, ads, ads_atom):
	
		# Expand atoms to be sure to acount for periodic repetition
		self.atoms_orig = atoms
		atoms_3x3 = atoms.repeat((3,3,1))
		
		super().__init__(atoms_3x3)
		self.ads = ads
		self.ads_atom = ads_atom
		
		# Get elements in the adsorbate
		self.ads_elems = re.findall('[A-Z][^A-Z]*', ads)
		
		# Get the index of the central adsorbing atom
		self.get_ads_atom_idx()
	
	def get_ads_atom_idx(self):
		'Return index of adsorbing atom on the 3x3 surface and update instance variable'
		ads_atom_ids = np.nonzero(self.symbols == self.ads_atom)[0]
		self.ads_atom_idx = ads_atom_ids[(len(ads_atom_ids)-1)//2]
		return self.ads_atom_idx
	
	def get_mean_location(self, atom_ids):
		'Return the average location of the specified atom indices'
		return np.mean(self.positions[atom_ids], axis=0)
	
	def get_distances_from_index(self, atom_idx, layer=None):
		'Return distances from all atoms of the given atom index restricted to a given layer if specified'
		if layer is None:
			return np.sum((self.positions - self.positions[atom_idx])**2, axis=1)**0.5
		else:
			positions = np.array([self.positions[atom.index] for atom in self if atom.tag==layer])
			return np.sum((positions - self.positions[atom_idx])**2, axis=1)**0.5
	
	def get_distances_from_point(self, point, layer=None):
		'Return distances from all atoms of the given atom index'
		if layer is None:
			return np.sum((self.positions - point)**2, axis=1)**0.5
		else:
			positions = np.array([self.positions[atom.index] for atom in self if atom.tag==layer])
			return np.sum((positions - point)**2, axis=1)**0.5
	
	def get_hollow_site(self, radius=2.6, hollow_radius=2.6, site_ids=None):
		'Return the elements and the type (fcc or hcp) of the hollow site,'
		'enforcing classification as a hollow site'
		
		if site_ids is None:
			
			# Get distances to all atoms
			dists = self.get_distances_from_index(self.ads_atom_idx)
			
			# Get atom indices below adsorbing atom
			mask_below = self.positions[:, -1] < self.positions[self.ads_atom_idx, -1]
			ids_below = np.array(list(range(len(self))))[mask_below]
			dists_below = dists[mask_below]
			
			# Get atom indices of the three closest atoms below the adsorbing atom
			sort_ids = np.argpartition(dists_below, kth=3)[:3]
			site_ids = ids_below[sort_ids]
		
		# Check that one of the angles that the three atoms form
		# is reasonably close to 60 degrees (this is to avoid classifying
		# on-top sites with three closest atoms on a line as a hollow site)
		vec1 = self.positions[site_ids[1], :2] - self.positions[site_ids[0], :2]
		len1 = sum(vec1**2)**0.5
		vec2 = self.positions[site_ids[2], :2] - self.positions[site_ids[0], :2]
		len2 = sum(vec2**2)**0.5
		dotp = np.dot(vec1, vec2)
		angle = np.arccos(dotp / (len1*len2)) * 180./np.pi
		
		# If the three atoms do not resemble a hollow site,
		# then it is not reasonable to classify as such
		if not (45. < angle < 75.):
			
			# Print a warning to help resolving this issue when it arises
			print(f'[WARNING] Hollow site with an angle of {angle} degrees far from 60-60-60 degrees detected')
			return None, None
		
		# Get center coordinates of site
		site_pos = self.get_mean_location(site_ids)
		
		# Make sure that site center is below the lowest atom
		site_pos[-1] = min(self.positions[site_ids, -1]) - 1.5
		
		# Check how many atoms are within a radius below
		# the center of the hollow site
		dists = self.get_distances_from_point(site_pos)
		ids_below = np.nonzero((dists < hollow_radius) * (self.positions[:, -1] < site_pos[-1]))[0]
		n_atoms_below = len(ids_below)
		
		# Get elements in site
		site_elems = self.symbols[site_ids]
		
		if n_atoms_below == 1:
			return site_elems, 'hcp'
		elif n_atoms_below == 3:
			return site_elems, 'fcc'
		else:
			view(self)
			raise ValueError(f'Number of atoms found under hollow site is {n_atoms_below} which is not recognized')
	
	def get_adsorption_site_ids(self, radius=2.6, hollow_radius=2.6):
		'Return the atom indices of the adsorption site'
		
		# Get distances from adsorbate atom to all other atoms
		dists = self.get_distances_from_index(self.ads_atom_idx)
		
		# Get atom indices coordinated to the adsorbate atom as the atoms within a sphere
		return np.nonzero((dists < radius) * (self.positions[:, -1] < self.positions[self.ads_atom_idx, -1]))[0]
	
	def get_adsorption_site(self, radius=2.6, hollow_radius=2.6):
		'Return the elements of the adsorption site as well as the site type'
		
		# Get atom indices of adsorption site atoms
		site_ids = self.get_adsorption_site_ids(radius, hollow_radius)
		
		# Get element symbols within the sphere, ignoring symbols in the adsorbate
		site_elems = sorted([self.symbols[idx] for idx in site_ids if self.symbols[idx] not in self.ads_elems])
		n_atoms_site = len(site_elems)
		
		if n_atoms_site == 1:
			return site_elems, 'ontop'
		elif n_atoms_site == 2:
			return site_elems, 'bridge'
		elif n_atoms_site == 3:
			site_elems, site = self.get_hollow_site(radius, hollow_radius, site_ids)
			
			# Raise an error if this is not a proper hollow site
			if None in [site_elems, site]:
				raise ValueError
			
			return site_elems, site
		else:
			raise ValueError(f'Number of atoms in site is {n_atoms_site} which is not recognized')
	
	def get_closest_ids(self, point, start, stop, layer):
		'Return the (stop - start + 1) closest atom indices from the given point in the specified layer'
		'E.g. start=1, stop=6, layer=1 will return the 6 closest atom indices in the surface'
		'layer starting from the closest atom and stopping at the 6th closest atom.'
		
		# Get distances to all atoms in the specified layer
		layer_atom_ids = np.array([atom.index for atom in self if atom.tag == layer])
		dists = self.get_distances_from_point(point, layer)
		
		# Get sorting ids
		ids_sort = np.argpartition(dists, kth=range(start, stop+1))[start:stop+1]
		
		# Get closest atom indices
		return layer_atom_ids[ids_sort]	
	
	def original_to_3x3_index(self, indices):
		'Return the atom indices of the central piece of teh 3x3 slab that corresponds'
		'to the specified atom indices in the original atoms object'
		return np.asarray(indices) + 4*len(self.atoms_orig)
		
	def view(self):
		view(self)
