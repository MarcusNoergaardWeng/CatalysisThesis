import sys
import xgboost;
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition
import numpy as np
import pandas as pd
import xgboost as xgb
import time
import random
#from skopt import gp_minimize
from ase.db import connect

sys.path.append('../scripts')
from Slab import expand_triangle, Slab, inside_triangle
from FeatureReader import OntopStandard111, FccStandard111

#### KEY VALUES ####
dim_x, dim_y = 200, 200
metals = ['Ag', 'Au', 'Cu', 'Pd', 'Pt']

metal_colors = dict(Pt = '#babbcb',
                    Pd = '#1f8685',
                    Ag = '#c3cdd6',
                    Cu = '#B87333',
                    Au = '#fdd766')

# Free (eV)
CO2   = {"ZPE": 0.31, "CpdT": 0.10, "minusTS": -0.66, "TS": 0.66}
CO    = {"ZPE": 0.13, "CpdT": 0.09, "minusTS": -0.61, "TS": 0.61}
H2    = {"ZPE": 0.28, "CpdT": 0.09, "minusTS": -0.40, "TS": 0.40}
H2O   = {"ZPE": 0.57, "CpdT": 0.10, "minusTS": -0.67, "TS": 0.67}
HCOOH = {"ZPE": 0.90, "CpdT": 0.11, "minusTS": -0.99, "TS": 0.99}
#Slab  = {"ZPE": 0.00, "CpdT": 0.00, "minusTS": -0.00} #Holy moly, den her overskrev Slab funktionen

# *Bound to the surface (eV)
# Bidentate *OOCH?
bound_CO   = {"ZPE": 0.19, "CpdT": 0.08, "minusTS": -0.16, "TS": 0.16}
bound_OH   = {"ZPE": 0.36, "CpdT": 0.05, "minusTS": -0.08, "TS": 0.08}
bound_OCHO = {"ZPE": 0.62, "CpdT": 0.11, "minusTS": -0.24, "TS": 0.24} #Either bidentate or monodentate. Use for both for now
bound_O    = {"ZPE": 0.07, "CpdT": 0.03, "minusTS": -0.04, "TS": 0.04}
bound_COOH = {"ZPE": 0.62, "CpdT": 0.10, "minusTS": -0.19, "TS": 0.19}
bound_H    = {"ZPE": 0.23, "CpdT": 0.01, "minusTS": -0.01, "TS": 0.01}

# Approximation Factors (FA)
AF = {"CO2": CO2, "CO": CO, "H2": H2, "H2O": H2O, "HCOOH": HCOOH, \
      "bound_CO": bound_CO, "bound_OH": bound_OH, "bound_OCHO": bound_OCHO, \
      "bound_O": bound_O, "bound_COOH": bound_COOH, "bound_H": bound_H}

# This is from the molecules_out.db file
molecules_dict = {'CO': -12.848598765234707,\
 'CO2': -19.15168636258064,\
 'CH2O2': -25.7548327798152,\
 'C2H4O2': -41.95993780269195,\
 'H2': -6.67878491734772,\
 'H2O': -12.225511685485456,\
 'CH2O': -19.92286258910958,\
 'CH4O': -27.652189372849637,\
 'C2H6O': -43.67355392866396,\
 'C2H2O2': -32.92328015484662,\
 'C2H2O4': -44.117581976029946}

#### MAKING FEATURES FROM DFT DATA ####

def features_from_DFT_data_H_and_COOH(features_folder, db_folder, db_name_H, db_name_COOH, db_name_slab, feature_file_H, feature_file_COOH):
    # Specify metals
    metals = ['Ag', 'Au', 'Cu', 'Pd', 'Pt']
    alloy = ''.join(metals)

    # Initiate feature readers
    reader_H = FccStandard111(metals)
    reader_COOH = OntopStandard111(metals)

    site_ids_H = [16, 17, 18]

    # Initiate counters of rejected samples
    rejected_H = 0
    rejected_COOH = 0

    # Writer headers to files
    with open(f'{features_folder}{feature_file_H}', 'w') as file_H:
        column_names = [f"feature{n}" for n in range(55)]
        column_names.append('G_ads (eV)')
        column_names.append('slab db row')
        column_names.append(f'{db_name_H}row')
        file_H.write(",".join(column_names))
        
    # Writer headers to files
    with open(f'{features_folder}{feature_file_COOH}', 'w') as file_COOH:
        column_names = [f"feature{n}" for n in range(20)]
        column_names.append('G_ads (eV)')
        column_names.append('slab db row')
        column_names.append(f'{db_name_COOH}row')
        file_COOH.write(",".join(column_names))
        
    # Load HEA(111) or Swim rings databases
    with connect(f'{db_folder}{db_name_H}') as db_H,\
        connect(f'{db_folder}{db_name_COOH}') as db_COOH,\
        connect(f'{db_folder}{db_name_slab}') as db_slab,\
        open(f'{features_folder}{feature_file_H}', 'a') as file_H,\
        open(f'{features_folder}{feature_file_COOH}', 'a') as file_COOH:
        #print("A1")
        # Iterate through slabs without adsorbates
        for row_slab in db_slab.select('energy', H=0, C=0, O=0): # This doesn't even trigger lmao
            #print("A2")
            # Iterate through the two adsorbates
            for ads in ['COOH', 'H']:
                #print("A3")
                # Set adsorbate-specific parameters
                if ads == 'COOH':
                    db = db_COOH
                    kw = {'C': 1,'O': 2, 'H': 1}
                    db_name = db_name_COOH
                    out_file = file_COOH

                elif ads == 'H':
                    db = db_H
                    kw = {'O': 0, 'H': 1}
                    db_name = db_name_H
                    out_file = file_H
                    ads_atom = "H"

                # Set counter of matched slabs between the databases to zero
                n_matched = 0

                # Get the corresponding slab with adsorbate
                for row in db.select('energy', **kw, **row_slab.count_atoms()):
                    #print("A4")
                    # If symbols match up
                    if row.symbols[:-len(ads)] == row_slab.symbols:
                        #print("A5")
                        # Increment the counter of matched structures
                        n_matched += 1

                        # Get atoms object
                        atoms = db.get_atoms(row.id)

                        # If the adsorbate is *COOH
                        if ads == 'COOH':
                            # Make slab instance
                            slab = Slab(atoms, ads=ads, ads_atom='C')

                            # Get adsorption site elements as neighbors within a radius
                            site_elems, site = slab.get_adsorption_site(radius=2.6, hollow_radius=2.6)

                            # If the site does not consist of exactly one atom, then skip this sample
                            # as the *OH has moved too far away from an on-top site
                            try:
                                if len(site_elems) !=1:
                                    rejected_COOH += 1
                                    #slab.view()
                                    continue
                            except TypeError:
                                print(site_elems, site)
                                print(row_slab.id, row.id)
                                slab.view()
                                exit()

                            # Get features of structure
                            features = reader_COOH.get_features(slab, radius=2.6)
                            
                            ### This part is now adsorbate-specific ###
                            # Get adsorption energy
                            E_ads = correct_DFT_energy_COOH(molecules_dict, row.energy, row_slab.energy) # This is the new formula

                            # Write output to file
                            features = ','.join(map(str, features))
                            out_file.write(f'\n{features},{E_ads:.6f},{row_slab.id},{row.id}')

                        # Else, if the adsorbate is H*
                        elif ads == 'H':
                            
                            atoms = atoms.repeat((3, 3, 1))
                            slab = Slab(atoms, ads=ads, ads_atom=ads_atom)
                            chemical_symbols = atoms.get_chemical_symbols()
                            #view(atoms)
                            H_index = [i for i, x in enumerate(chemical_symbols) if x == "H"][4]
                            
                            all_distances = atoms.get_distances([n for n in list(range(len(chemical_symbols))) if n != H_index], H_index)
                            site_ids_H = np.argpartition(all_distances, 2)[0:3]
                            site_ids_H = [x+1 if x>229 else x for x in site_ids_H] #Compensates for the removal of an H, so that the indices above 229 are not one too small
                            #print("site_ids_H: ", site_ids_H)
                            # Get hollow site planar corner coordinates
                            site_atoms_pos_orig = atoms.positions[site_ids_H, :2]

                            # Get expanded triangle vertices
                            site_atoms_pos = expand_triangle(site_atoms_pos_orig, expansion=1.45)

                            # Get position of adsorbate atom (with atom index XXX 20 XXX)
                            ads_pos = atoms.positions[H_index][:2]

                            # If the H is outside the expanded fcc triangle,
                            # then it is most likely in an hcp site, that is not
                            # being modeled
                            if not inside_triangle(ads_pos, site_atoms_pos):
                                rejected_H += 1
                                continue

                            # Get features of structure
                            features = reader_H.get_features(slab, radius=2.6, site_ids=site_ids_H)

                            ### This part is now adsorbate-specific ###
                            # Get adsorption energy
                            E_ads = correct_DFT_energy_H(molecules_dict, row.energy, row_slab.energy) # This is the new formula
                            
                            # Write output to file
                            features = ','.join(map(str, features))
                            out_file.write(f'\n{features},{E_ads:.6f},{row_slab.id},{row.id}')

                if n_matched > 1:
                    print(f'[INFO] {n_matched} {ads} and slab matched for row {row_slab.id} in {db_name_slab}')

    # Print the number of rejected samples to screen
    print('rejected COOH samples: ', rejected_COOH)
    print('rejected H samples: ', rejected_H)
    return None

#### CORRECTION CONSTANTS AND CORRECTING DFT ENERGIES ####

def calc_correction_constant_COOH(AF):
    ### Summing up all the Approximation Factors for COOH
    ZpE_sum  = AF["bound_COOH"]["ZPE"]  - AF["HCOOH"]["ZPE"]  + 1/2*AF["H2"]["ZPE"]
    CpdT_sum = AF["bound_COOH"]["CpdT"] - AF["HCOOH"]["CpdT"] + 1/2*AF["H2"]["CpdT"]
    TS_sum   = AF["bound_COOH"]["TS"]   - AF["HCOOH"]["TS"]   + 1/2*AF["H2"]["TS"] #Figure the signs out
    correction_constant_COOH = ZpE_sum + CpdT_sum - TS_sum
    return correction_constant_COOH

def correct_DFT_energy_COOH(molecules_dict, E_COOH, E_slab):
    DeltaE = E_COOH - molecules_dict["CH2O2"] + 1/2*molecules_dict["H2"] - E_slab
    correction_constant_COOH = calc_correction_constant_COOH(AF)
    DeltaG = DeltaE + correction_constant_COOH
    return DeltaG

def calc_correction_constant_H(AF):
    ### Summing up all the Approximation Factors for H
    ZpE_sum  = AF["bound_H"]["ZPE"]  - 1/2 * AF["H2"]["ZPE"] # Stokiometrien er bevaret! Tak, Oliver
    CpdT_sum = AF["bound_H"]["CpdT"] - 1/2 * AF["H2"]["CpdT"]
    TS_sum   = AF["bound_H"]["TS"]   - 1/2 * AF["H2"]["TS"]
    correction_constant_H = ZpE_sum + CpdT_sum - TS_sum
    return correction_constant_H

def correct_DFT_energy_H(molecules_dict, E_H, E_slab):
    DeltaE = E_H - 1/2*molecules_dict["H2"] - E_slab
    correction_constant_H = calc_correction_constant_H(AF)
    DeltaG = DeltaE + correction_constant_H
    return DeltaG

#### MAKE SWIM RING SURFACE ####

def AB_to_split(A, B): #Tested, works
    split = [0, 0, 0, 0, 0]

    if "Ag" in B:
        split[0] += 6/len(B)
    if "Au" in B:
        split[1] += 6/len(B)
    if "Pd" in A:
        split[3] += 1/len(A)
    if "Pt" in A:
        split[4] += 1/len(A)
    return split

def make_long_vector(A, B, vector_length): # Tested, works
    long_vector = [random.choice(A) for _ in range(vector_length // 3) for _ in range(3)]
    long_vector[1::3] = [random.choice(B) for _ in range(vector_length // 3)]
    long_vector[2::3] = [random.choice(B) for _ in range(vector_length // 3)]
    return long_vector

def initialize_swim_surface(A, B): # Tested, works
    # Make the split from the metals used
    split = AB_to_split(A, B)

    # Make the surface (empty surface)
    surface = initialize_surface(dim_x, dim_y, metals, split)

    # Change out the top layer for the optimal swim ring surface
    #n = 4*35
    #top_dim_x, top_dim_y = 3*n, 3*n-1
    #n_adams = int(top_dim_x * top_dim_y / 3)

    # I need to know the closest number of atoms above the min amount that is divisible by 3
    n_adams = dim_x*dim_y
    long_vector = make_long_vector(A, B, n_adams+2)[0:n_adams] #Den lange vector skal ende med at være dim_x*dim_y lang

    top_layer = np.reshape(long_vector, (dim_x, dim_y))
    top_layer = top_layer[0:dim_x, 0:dim_y]

    # Set the neatly arranged layer
    surface["atoms"][:,:,0] = top_layer

    # Predict energies on all sites for both adsorbates
    surface = precompute_binding_energies_SPEED(surface, dim_x, dim_y, models)
    return surface

#### TRAINING BINDING ENERGY PREDICTION MODELS ####



#### LOAD BINDING ENERGY MODELS ####

def load_G_models():
    #Load H binding energy (G) prediction model
    H_DFT_model = xgb.Booster({'nthread': 8})
    H_DFT_model.load_model("../models/"+"H_DFT.model")

    #Load COOH binding energy (G) prediction model
    COOH_DFT_model = xgb.Booster({'nthread': 8})
    COOH_DFT_model.load_model("../models/"+"COOH_DFT.model")

    #Load mixed site energy (G)  prediction model
    mixed_DFT_model = xgb.Booster({'nthread': 8})
    mixed_DFT_model.load_model("../models/"+"COOH_H.model")

    models = {"H": H_DFT_model, "COOH": COOH_DFT_model, "mixed": mixed_DFT_model}
    return models

models = load_G_models() # I think I have to load it in here in order for the bayesian optimization scheme to work. I can't pass stuff to the functions

#### BAYESIAN OPTIMIZATION ROUTINE ####

def Bayesian_optimization(space, simulate_loss_type):
    ## Normalize the search space dimensions
    #space = normalize_dimensions(space)

    # Initialize the Bayesian Optimizer
    optimizer = gp_minimize(
        simulate_loss_type,
        space,
        n_calls = 50, # Number of evaluations of the loss function
        random_state=42, # Set a random seed for reproducibility
        n_jobs = 1)

    # Retrieve the intermediate results
    results = optimizer.x_iters
    losses = optimizer.func_vals

    # Print the intermediate results
    for i, result in enumerate(results):
        loss = losses[i]
        print(f"Iteration {i+1}: Surface Stochiometry: {result}, Loss: {loss}")

    # Retrieve the optimal solution
    optimal_surface_stochiometry = optimizer.x
    optimal_loss = optimizer.fun  # Negate the loss to retrieve the maximized value

    print("Optimal Surface Stochiometry:", optimal_surface_stochiometry)
    print("Optimal Loss:", optimal_loss)
    return results, optimal_surface_stochiometry

#### PLOT THE RESULTS FROM THE BAYESIAN OPTIMIZATION ####

def Bayesian_optimization_plot(results, losses, experiment_name):
    # Plain colors list
    colors_list = [metal_colors[metal] for metal in metals]

    # Plot the progression of surface stochiometries #ChatGPT
    fig = plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    for i, stoch in enumerate(np.array([stoch/np.sum(stoch) for stoch in results]).T):
        plt.plot(stoch, c = colors_list[i], label = metals[i]) # HERE - Trying to make it plot each line in the appropriate colors
    plt.title('Progression of Surface Stochiometries')
    plt.xlabel('Iteration')
    plt.ylabel('Surface Stochiometry')
    plt.legend(loc="upper right", ncol = 2)
    #plt.xlim(-5, np.shape(results)[0]*1.05)

    # Plot the losses as a function of iteration
    plt.subplot(2, 1, 2)
    plt.plot(-losses)
    plt.title('Number of good sites at each iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Good sites')

    # Adjust the layout of the subplots
    plt.tight_layout()

    # Save figure
    #experiment_name = "140by140_100its_diagonal"
    plt.savefig("../figures/DFT_calc_energies/"+experiment_name+"_progression.png", dpi = 300, bbox_inches = "tight")

    # Show the plot
    plt.show()
    return None

#### PLOTTING THE METALS IN THE GOOD HOLLOW SITES ####

def metals_in_good_hollow_sites_plot(surface, reward_type, experiment_name):

    import matplotlib.gridspec as gridspec

    E_top_dict, E_hol_dict, good_hol_sites, n_ratios = sort_energies(surface, reward_type)
    # Create the figure and gridspec
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[6, 3])

    # Plot on the left subplot
    good_hol_sites_sorted = sorted(["".join(sorted(x, reverse = True)) for x in good_hol_sites])
    good_hol_sites_dict = Counter(good_hol_sites_sorted)

    labels = good_hol_sites_dict.keys()
    values = good_hol_sites_dict.values()

    ax1 = plt.subplot(gs[0])
    ax1.bar(labels, values)
    ax1.set_xticklabels(labels, rotation=45)
    ax1.set_xlabel('Combinations')
    ax1.set_ylabel('Count')
    ax1.set_title('Frequency of combinations of metals in good hollow sites')

    # Plot on the right subplot
    ax2 = plt.subplot(gs[1])
    good_hol_sites_flat_dict = Counter(np.array(good_hol_sites).flatten())
    labels = good_hol_sites_flat_dict.keys()
    values = good_hol_sites_flat_dict.values()

    colors = [metal_colors[metal] for metal in labels]

    ax2.bar(labels, values, color = colors)
    ax2.set_xlabel('Metals')
    ax2.set_ylabel('Count')
    ax2.set_title('Frequency of each metal')

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.3)

    plt.savefig("../figures/DFT_calc_energies/"+experiment_name+"_good_sites.png", dpi = 300, bbox_inches = "tight")
    # Show the plot
    plt.show()
    return None

#### FUNCTIONS FOR PLOTTING BINDING ENERGIES H VS COOH ####

def plot_pure_metals(SE_slab_metals, DeltaG_H, DeltaG_COOH, metal_colors):
    # Create a figure and axes
    fig, ax = plt.subplots(figsize = (6, 6))

    # Set the limits for both x and y axes
    ax.set_xlim(-0.6, 1.1)
    ax.set_ylim(-0.6, 1.1)

    ## Set the major ticks and tick labels
    #ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
    #ax.set_xticklabels([-0.5, '', 0, '', 0.5, '', 1.0])
    #ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
    #ax.set_yticklabels([-0.5, '', 0, '', 0.5, '', 1.0])

    # Set the major ticks and tick labels
    ax.set_xticks([-0.5, 0, 0.5, 1.0])
    ax.set_xticklabels([-0.5, 0, 0.5, 1.0])
    ax.set_yticks([-0.5, 0, 0.5, 1.0])
    ax.set_yticklabels([-0.5, 0, 0.5, 1.0])

    # Set the grid lines
    ax.grid(which='both', linestyle=':', linewidth=0.5, color='gray')

    ax.set_title("DFT calculated energies for single metals fcc(111)")
    ax.set_xlabel("$\Delta G_{^*H}$ [eV]")
    ax.set_ylabel("$\Delta G_{^*COOH}$ [eV]")

    for i, metal in enumerate(SE_slab_metals):
            ax.scatter(DeltaG_H[i], DeltaG_COOH[i], label = "Pure "+metal, marker = "o", c = metal_colors[metal], edgecolors='black')
        
    if True:
        # Create a rectangle patch
        rect = patches.Rectangle((0, 0), 1.1, -0.6, linewidth=1, edgecolor='none', facecolor='green', alpha = 0.1)

        # Add the rectangle to the plot
        ax.add_patch(rect)
        
        label_text = f'Optimal area' #This can show how many points are in there as well
        label_x = 0.55
        label_y = -0.3
        ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=12)

    ax.legend()
    plt.savefig("../figures/DFT_calc_energies/"+"Single_Metals"+".png", dpi = 600, bbox_inches = "tight")
    plt.show()
    return None

def sort_energies(surface, reward_type): # Just make it return everything all the time
    
    # So this could quite nicely be a dictionary
    
    E_top_dict = {"Pt": [], "Pd": [], "Cu": [], "Ag": [], "Au": []}
    E_hol_dict = {"Pt": [], "Pd": [], "Cu": [], "Ag": [], "Au": []} # Hollow sites that have a neighbour of that atom type and their index match up with the adsorbate on the top-site of that atom.
    
    # Which atoms are around the GOOD hollow sites?
    good_hol_sites = []
    n_sites = 0 # All sites
    n_right_corner = 0  # The good sites
    n_left_corner  = 0  # The bad  sites
    n_diagonal     = 0  # The diagonal sites
    
    for x_top, y_top in [(x, y) for x in range(dim_x) for y in range(dim_y)]:
        for x_diff, y_diff in [(0, 0), (0, -1), (-1, 0)]:                     
            n_sites += 1
            # What are the indices?
            x_hollow = (x_top + x_diff) % dim_x
            y_hollow = (y_top + y_diff) % dim_y
            
            # What are the energies?
            on_top_E = surface["COOH_G"][x_top][y_top]
            hollow_E = surface["H_G"][x_hollow][y_hollow]
            
            # Which atom is the top-site?
            top_site_atom = surface["atoms"][x_top, y_top, 0]
            
            # Append the information to the dicts and lists
            E_top_dict[top_site_atom].append(on_top_E)
            E_hol_dict[top_site_atom].append(hollow_E)
            
            # Find GOOD sites:
            if (on_top_E < 0) and (hollow_E > 0): # Wait have a think about the loops
                # Here is a good site!
                n_right_corner += 1
                # I am interested in knowing which metals are around the hollow sites except for Pt
                atom1 = surface["atoms"][(x_hollow+0)%dim_x, (y_hollow+0)%dim_y, 0]
                atom2 = surface["atoms"][(x_hollow+1)%dim_x, (y_hollow+0)%dim_y, 0]
                atom3 = surface["atoms"][(x_hollow+0)%dim_x, (y_hollow+1)%dim_y, 0]
                good_hol_sites.append([atom1, atom2, atom3])

            if on_top_E < hollow_E: # The on-top binding energy is lower than hollow binding energy. Smaller means binds better
                # Here is a good site!
                n_diagonal += 1

            if on_top_E < 0 and hollow_E < 0:
                #Here is a bad site
                n_left_corner += 1
            
            
    n_left_corner_ratio = n_left_corner / n_sites
    n_right_corner_ratio = n_right_corner / n_sites
    n_diagonal_ratio = n_diagonal / n_sites
    n_ratios = {"left_corner": n_left_corner_ratio, "right_corner": n_right_corner_ratio, "diagonal": n_diagonal_ratio}
    return E_top_dict, E_hol_dict, good_hol_sites, n_ratios

# Make this into a function! And make it save a nice plot. Perhaps ask for a name directly in the function-call
def deltaEdeltaE_plot(filename, surface, pure_metal_info, reward_type, show_plot):

    # First, calculate the statistics of interest
    E_top_dict, E_hol_dict, good_hol_sites, n_ratios = sort_energies(surface, reward_type)

    fig, ax = plt.subplots(figsize = (6, 6))
    
    # Set the limits for both x and y axes
    ax.set_xlim(-0.6, 1.1)
    ax.set_ylim(-0.6, 1.1)
    
    # Set the major ticks and tick labels
    ax.set_xticks([-0.5, 0, 0.5, 1.0])
    ax.set_xticklabels([-0.5, 0, 0.5, 1.0])
    ax.set_yticks([-0.5, 0, 0.5, 1.0])
    ax.set_yticklabels([-0.5, 0, 0.5, 1.0])
    
    # Set the grid lines
    ax.grid(which='both', linestyle=':', linewidth=0.5, color='gray')
    
    ax.set_title("Predicted energies for $^*COOH$ and $^*H$ for whole surface")
    ax.set_xlabel("$\Delta G_{^*H}$ [eV]")
    ax.set_ylabel("$\Delta G_{^*COOH}$ [eV]")
    
    if reward_type == "right_corner":
        # Create a rectangle patch
        rect = patches.Rectangle((0, 0), 1.1, -0.6, linewidth=1, edgecolor='none', facecolor='green', alpha = 0.1)
    
        # Add the rectangle to the plot
        ax.add_patch(rect)
        
        label_text = f'Optimal area \n$n_{{optimal}} / n_{{sites}} = {100*n_ratios[reward_type]:.2f} \%$' #This can show how many points are in there as well
        label_x = 0.55
        label_y = -0.3
        ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=12)
    
    if reward_type == "diagonal":
        ax.plot([-0.6, 1.5], [-0.6, 1.5], 'k--', alpha = 0.0)  # Plot the diagonal line
        ax.fill_between([-0.6, 1.5], [-0.6, 1.5], -1, where=(y >= -1), color='green', alpha=0.1)  # Fill the area under the line

        label_text = f'Diagonal area \n$n_{{diagonal}} / n_{{sites}} = {100*n_ratios[reward_type]:.2f} \%$' #This can show how many points are in there as well
        label_x = 0.55
        label_y = -0.3
        ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=12)

    if reward_type == "left_corner":
        # Create a rectangle patch
        rect = patches.Rectangle((-0.6, -0.6), 0.6, 0.6, linewidth=1, edgecolor='none', facecolor='red', alpha = 0.1)
    
        # Add the rectangle to the plot
        ax.add_patch(rect)
        
        label_text = f'Danger zone \n$n_{{bad}} / n_{{sites}} = {100*n_ratios[reward_type]:.2f} \%$' #This can show how many points are in there as well
        label_x = -0.3
        label_y = -0.3
        ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=12)

    if reward_type == "both_corners":
        # Create a rectangle patch
        rect = patches.Rectangle((-0.6, -0.6), 0.6, 0.6, linewidth=1, edgecolor='none', facecolor='red', alpha = 0.1)
    
        # Add the rectangle to the plot
        ax.add_patch(rect)
        
        label_text = f'Danger zone \n$n_{{bad}} / n_{{sites}} = {100*n_ratios["left_corner"]:.2f} \%$' #This can show how many points are in there as well
        label_x = -0.3
        label_y = -0.3
        ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=12)

        # Create a rectangle patch
        rect = patches.Rectangle((0, 0), 1.1, -0.6, linewidth=1, edgecolor='none', facecolor='green', alpha = 0.1)
    
        # Add the rectangle to the plot
        ax.add_patch(rect)
        
        label_text = f'Optimal area \n$n_{{good}} / n_{{sites}} = {100*n_ratios["right_corner"]:.2f} \%$' #This can show how many points are in there as well
        label_x = 0.55
        label_y = -0.3
        ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=12)

    if reward_type == "opt_line":
        ax.axhline(y=-0.12, color='black', linestyle='solid')
        # Add a text label without LaTeX rendering
        label_text = r'$\Delta G_{\mathrm{opt}} = -0.12\ \mathrm{eV}$'
        label_x = 0.8
        label_y = -0.09
        plt.text(label_x, label_y, label_text, ha='center', va='center', fontsize=10)

    stochiometry = surface["stochiometry"]
    for metal in ['Ag', 'Au', 'Cu', 'Pd', 'Pt']:
        ax.scatter(E_hol_dict[metal], E_top_dict[metal], label = f"{metal}$_{{{stochiometry[metal]:.2f}}}$", s = 0.5, alpha = 0.8, c = metal_colors[metal]) # edgecolor = "black", linewidth = 0.05, 
    
    if True:
        for i, metal in enumerate(pure_metal_info["SE_slab_metals"]):
            ax.scatter(pure_metal_info["DeltaG_H"][i], pure_metal_info["DeltaG_COOH"][i], label = "Pure "+metal, marker = "o", c = metal_colors[metal], edgecolors='black')
        
    ax.legend(loc="upper right")
    
    plt.savefig("../figures/DFT_calc_energies/"+filename+".png", dpi = 600, bbox_inches = "tight")
    if show_plot == True:
        plt.show()
    else:
        plt.close()
    return None

#### FUNCTIONS FOR BAYESIAN OPTIMIZATION ####

def simulate_loss_right_corner(surface_stochiometry):
    ## surface_stochiometry is a 5-len list of probabilities to draw each metal
    surface_stochiometry = np.array(surface_stochiometry) / np.sum(surface_stochiometry)
    metals = ['Ag', 'Au', 'Cu', 'Pd', 'Pt']
    surface = initialize_surface(dim_x, dim_y, metals, surface_stochiometry)
    
    # Predict energies on all sites for both adsorbates
    surface = precompute_binding_energies_SPEED(surface, dim_x, dim_y, models)
    #n_sites = 0
    n_good  = 0
    for x_top, y_top in [(x, y) for x in range(dim_x) for y in range(dim_y)]: # Mixed order
        for x_diff, y_diff in [(0, 0), (0, -1), (-1, 0)]:                     # Mixed order
            #n_sites += 1
            # What are the indices?
            x_hollow = (x_top + x_diff) % dim_x
            y_hollow = (y_top + y_diff) % dim_y
            
            # What are the energies?
            on_top_E = surface["COOH_G"][x_top][y_top]
            hollow_E = surface["H_G"][x_hollow][y_hollow]
            
            # Find GOOD sites:
            if (on_top_E < 0) and (hollow_E > 0):
                # Here is a good site!
                n_good += 1
    return -n_good

def simulate_loss_left_corner(surface_stochiometry):
    ## surface_stochiometry is a 5-len list of probabilities to draw each metal
    surface_stochiometry = np.array(surface_stochiometry) / np.sum(surface_stochiometry)
    metals = ['Ag', 'Au', 'Cu', 'Pd', 'Pt']
    surface = initialize_surface(dim_x, dim_y, metals, surface_stochiometry)
    
    # Predict energies on all sites for both adsorbates
    surface = precompute_binding_energies_SPEED(surface, dim_x, dim_y, models)
    #n_sites = 0
    n_bad  = 0
    for x_top, y_top in [(x, y) for x in range(dim_x) for y in range(dim_y)]: # Mixed order
        for x_diff, y_diff in [(0, 0), (0, -1), (-1, 0)]:                     # Mixed order
            #n_sites += 1
            # What are the indices?
            x_hollow = (x_top + x_diff) % dim_x
            y_hollow = (y_top + y_diff) % dim_y
            
            # What are the energies?
            on_top_E = surface["COOH_G"][x_top][y_top]
            hollow_E = surface["H_G"][x_hollow][y_hollow]
            
            # Find BAD sites:
            if (on_top_E < 0) and (hollow_E < 0): # Low COOH and low H
                # Here is a bad site!
                n_bad += 1
    return n_bad

def simulate_loss_both_corners(surface_stochiometry):
    ## surface_stochiometry is a 5-len list of probabilities to draw each metal
    surface_stochiometry = np.array(surface_stochiometry) / np.sum(surface_stochiometry)
    metals = ['Ag', 'Au', 'Cu', 'Pd', 'Pt']
    surface = initialize_surface(dim_x, dim_y, metals, surface_stochiometry)
    
    # Predict energies on all sites for both adsorbates
    surface = precompute_binding_energies_SPEED(surface, dim_x, dim_y, models)
    #n_sites = 0
    n_loss  = 0
    for x_top, y_top in [(x, y) for x in range(dim_x) for y in range(dim_y)]: # Mixed order
        for x_diff, y_diff in [(0, 0), (0, -1), (-1, 0)]:                     # Mixed order
            #n_sites += 1
            # What are the indices?
            x_hollow = (x_top + x_diff) % dim_x
            y_hollow = (y_top + y_diff) % dim_y
            
            # What are the energies?
            on_top_E = surface["COOH_G"][x_top][y_top]
            hollow_E = surface["H_G"][x_hollow][y_hollow]
            
            # Find BAD sites:
            if (on_top_E < 0) and (hollow_E < 0): # Low COOH and low H
                # Here is a bad site!
                n_loss += 1

            # Find GOOD sites:
            if (on_top_E < 0) and (hollow_E > 0):
                # Here is a good site!
                n_loss -= 1
    return n_loss

## Make a loss function, that rewards points for being under the diagonal
def simulate_loss_diagonal(surface_stochiometry):
        ## surface_stochiometry is a 5-len list of probabilities to draw each metal
    surface_stochiometry = np.array(surface_stochiometry) / np.sum(surface_stochiometry)
    metals = ['Ag', 'Au', 'Cu', 'Pd', 'Pt']
    surface = initialize_surface(dim_x, dim_y, metals, surface_stochiometry)
    
    # Predict energies on all sites for both adsorbates
    surface = precompute_binding_energies_SPEED(surface, dim_x, dim_y, models)
    #n_sites = 0
    n_under_diagonal  = 0
    for x_top, y_top in [(x, y) for x in range(dim_x) for y in range(dim_y)]: # Mixed order
        for x_diff, y_diff in [(0, 0), (0, -1), (-1, 0)]:                     # Mixed order
            #n_sites += 1
            # What are the indices?
            x_hollow = (x_top + x_diff) % dim_x
            y_hollow = (y_top + y_diff) % dim_y
            
            # What are the energies?
            on_top_E = surface["COOH_G"][x_top][y_top]
            hollow_E = surface["H_G"][x_hollow][y_hollow]
            
            # Find GOOD sites:
            if on_top_E < hollow_E: # The on-top binding energy is lower than hollow binding energy. Smaller means binds better
                # Here is a good site!
                n_under_diagonal += 1

    return -n_under_diagonal

#### FUNCTIONS FOR PREDITING ENERGIES ####

def calc_given_energies(surface):
    surface["COOH_given_H_down"]     = surface["mixed_down"]     - np.roll(surface["H_G"], (-1,  0), axis=(0, 1))
    surface["COOH_given_H_up_right"] = surface["mixed_up_right"] - np.roll(surface["H_G"], ( 0,  0), axis=(0, 1))
    surface["COOH_given_H_up_left"]  = surface["mixed_up_left"]  - np.roll(surface["H_G"], ( 0, -1), axis=(0, 1))

    surface["H_given_COOH_down"]     = surface["mixed_down"]     - np.roll(surface["COOH_G"], (+1,  0), axis=(0, 1))
    surface["H_given_COOH_up_right"] = surface["mixed_up_right"] - np.roll(surface["COOH_G"], ( 0,  0), axis=(0, 1))
    surface["H_given_COOH_up_left"]  = surface["mixed_up_left"]  - np.roll(surface["COOH_G"], ( 0, +1), axis=(0, 1))

    return surface

def predict_mixed_energies(surface, dim_x, dim_y, models):
    COOH_down_features     = []
    COOH_up_right_features = []
    COOH_up_left_features  = []

    difs = {"down": {"x": 0, "y": -1}, "up_right": {"x": 0, "y": 0}, "up_left": {"x": -1, "y": 0}}
    
    # Make features for each site:
    for top_site_x, top_site_y in [(top_site_x, top_site_y) for top_site_x in range(dim_x) for top_site_y in range(dim_y)]:

        # Down 
        hol_site_x = top_site_x + difs["down"]["x"]
        hol_site_y = top_site_y + difs["down"]["y"]
        COOH_down_features.append(mixed_site_vector(surface["atoms"], hol_site_x, hol_site_y, top_site_x, top_site_y))


        # Up right
        hol_site_x = top_site_x + difs["up_right"]["x"]
        hol_site_y = top_site_y + difs["up_right"]["y"]
        COOH_up_right_features.append(mixed_site_vector(surface["atoms"], hol_site_x, hol_site_y, top_site_x, top_site_y))

        # Up left
        hol_site_x = top_site_x + difs["up_left"]["x"]
        hol_site_y = top_site_y + difs["up_left"]["y"]
        COOH_up_left_features.append(mixed_site_vector(surface["atoms"], hol_site_x, hol_site_y, top_site_x, top_site_y))

    # Remove the uneccesary singleton dimension
    COOH_down_features     = np.squeeze(COOH_down_features)
    COOH_up_right_features = np.squeeze(COOH_up_right_features)
    COOH_up_left_features  = np.squeeze(COOH_up_left_features)

    # Make the features into a big dataframe
    COOH_down_features_df     = pd.DataFrame(COOH_down_features     , columns = [f"feature{n}" for n in range(75)])
    COOH_up_right_features_df = pd.DataFrame(COOH_up_right_features , columns = [f"feature{n}" for n in range(75)])
    COOH_up_left_features_df  = pd.DataFrame(COOH_up_left_features  , columns = [f"feature{n}" for n in range(75)])

    # Turn them into DMatrix
    COOH_down_features_DM     = pandas_to_DMatrix(COOH_down_features_df)
    COOH_up_right_features_DM = pandas_to_DMatrix(COOH_up_right_features_df)
    COOH_up_left_features_DM  = pandas_to_DMatrix(COOH_up_left_features_df)
    
    # Predict energies in one long straight line
    
    COOH_down_G = models["mixed"].predict(COOH_down_features_DM)
    COOH_up_right_G = models["mixed"].predict(COOH_up_right_features_DM)
    COOH_up_left_G = models["mixed"].predict(COOH_up_left_features_DM)

    # Make them into a nice matrix shape - in a minute
    COOH_down_G = np.reshape(COOH_down_G, (dim_x, dim_y))
    COOH_up_right_G = np.reshape(COOH_up_right_G, (dim_x, dim_y))
    COOH_up_left_G = np.reshape(COOH_up_left_G, (dim_x, dim_y))
    
    # Attach the energies to the matrices in the surface dictionary

    surface["mixed_down"]     = COOH_down_G
    surface["mixed_up_right"] = COOH_up_right_G
    surface["mixed_up_left"]  = COOH_up_left_G

    surface = calc_given_energies(surface)
    return surface

def initialize_surface(dim_x, dim_y, metals, split): #Is still random - could be used with a seed in the name of reproduceability
    dim_z = 3
    
    surf_atoms = create_surface(dim_x, dim_y, metals, split)
    
    # Binding energies
    surf_COOH_G = np.reshape([np.nan]*dim_x*dim_y, (dim_x, dim_y))# On-top sites
    surf_H_G    = np.reshape([np.nan]*dim_x*dim_y, (dim_x, dim_y))# Hollow sites

    # Mixed-site energies
    surf_COOH_down     = np.reshape([np.nan]*dim_x*dim_y, (dim_x, dim_y))
    surf_COOH_up_right = np.reshape([np.nan]*dim_x*dim_y, (dim_x, dim_y))
    surf_COOH_up_left  = np.reshape([np.nan]*dim_x*dim_y, (dim_x, dim_y))

    stochiometry = dict(zip(metals, np.array(split)/np.sum(split)))
    
    surf = {"atoms": surf_atoms, "stochiometry": stochiometry,\
            "COOH_G": surf_COOH_G, "H_G": surf_H_G, "mixed_down": surf_COOH_down, "mixed_up_right": surf_COOH_up_right, "mixed_up_left": surf_COOH_up_left}
    return surf

def create_surface(dim_x, dim_y, metals, split):
    dim_z = 3
    num_atoms = dim_x*dim_y*dim_z
    if np.sum(split) != 1.0:
        # This split is not weighted properly, I'll fix it
        split = split / np.sum(split)
    if split == "Even":
        proba = [1.0 / len(metals) for n in range(len(metals))] 
        surface = np.random.choice(metals, num_atoms, p=proba)
    else:
        surface = np.random.choice(metals, num_atoms, p=split)
    surface = np.reshape(surface, (dim_x, dim_y, dim_z)) #Reshape list to the
    return surface

def precompute_binding_energies_TQDM(surface, dim_x, dim_y, models, predict_G_function): #TJEK I think this function can go faster if I make all the data first appended to a list, then to a PD and then 
    for x, y in tqdm([(x, y) for x in range(dim_x) for y in range(dim_y)], desc = r"Predicting all ΔG", leave = False): # I could randomise this, so I go through all sites in a random order
        
        ads = "H"
        surface["H_G"][x][y] = predict_G_function(surface["atoms"], x, y, ads, models) ## A new function that wraps/uses the XGBoost model
        
        ads = "COOH"
        surface["COOH_G"][x][y] = predict_G_function(surface["atoms"], x, y, ads, models) ## A new function that wraps/uses the XGBoost model

    return surface

def precompute_binding_energies_SPEED(surface, dim_x, dim_y, models):
    H_features    = []
    COOH_features = []
    #index_pairs   = []

    # Make features for each site:
    for x, y in [(x, y) for x in range(dim_x) for y in range(dim_y)]:#, desc = r"Making all feature vectors", leave = True): # I could randomise this, so I go through all sites in a random order
        # Append the features
        H_features.append([hollow_site_vector(surface["atoms"], x, y)])
        COOH_features.append([on_top_site_vector(surface["atoms"], x, y)])
        #index_pairs.append([str(x)+","+str(y)])

    # Remove the uneccesary singleton dimension
    H_features = np.squeeze(H_features)
    COOH_features = np.squeeze(COOH_features)

    # Make the features into a big dataframe
    H_features_df    = pd.DataFrame(H_features   , columns = [f"feature{n}" for n in range(55)])
    COOH_features_df = pd.DataFrame(COOH_features, columns = [f"feature{n}" for n in range(20)])

    # Turn them into DMatrix
    H_features_DM    = pandas_to_DMatrix(H_features_df)
    COOH_features_DM = pandas_to_DMatrix(COOH_features_df)

    # Predict energies in one long straight line
    H_G    = models["H"].predict(H_features_DM)
    COOH_G = models["COOH"].predict(COOH_features_DM)

    # Make them into a nice matrix shape - in a minute
    H_G    = np.reshape(H_G   , (dim_x, dim_y))
    COOH_G = np.reshape(COOH_G, (dim_x, dim_y))
    #index_pairs = np.reshape(index_pairs, (dim_x, dim_y))

    # Attach the energies to the matrices in the surface dictionary
    surface["H_G"]    = H_G
    surface["COOH_G"] = COOH_G

    # Predict the energies on the mixed sites
    surface = predict_mixed_energies(surface, dim_x, dim_y, models)

    # Calculate the "*COOH given *H" and "*H given *COOH" energies
    surface = calc_given_energies(surface)

    return surface

def predict_G(surface, site_x, site_y, adsorbate, models):
    if adsorbate == "H":
        vector_df = pd.DataFrame([hollow_site_vector(surface, site_x, site_y)], columns = [f"feature{n}" for n in range(55)])
        vector_DM = pandas_to_DMatrix(vector_df)
        G = models["H"].predict(vector_DM)[0]
        return G
    
    if adsorbate == "COOH":
        vector_df = pd.DataFrame([on_top_site_vector(surface, site_x, site_y)], columns = [f"feature{n}" for n in range(20)])
        vector_DM = pandas_to_DMatrix(vector_df)
        G = models["COOH"].predict(vector_DM)[0]
        return G
    
def on_top_site_vector(surface, site_x, site_y): # I should have done modulo to dim_x and dim_y
    dim_x, dim_y = np.shape(surface)[0], np.shape(surface)[1]
    site1 = [surface[site_x, site_y, 0]]# Make a one-hot encoded vector of the very site here! Add at the beginning 
    site1_count = [site1.count(metals[n]) for n in range(len(metals))]
    
    top6 = [surface[site_x % dim_x, (site_y-1) % dim_y, 0], surface[site_x % dim_x, (site_y+1) % dim_y, 0], surface[(site_x-1) % dim_x, site_y % dim_y, 0], surface[(site_x+1) % dim_x, site_y % dim_y, 0], surface[(site_x-1) % dim_x, (site_y+1) % dim_y, 0], surface[(site_x+1) % dim_x, (site_y-1) % dim_y, 0]]
    top6_count = [top6.count(metals[n]) for n in range(len(metals))]
    
    mid3 = [surface[(site_x-1) % dim_x, (site_y-1) % dim_y,1], surface[site_x % dim_x, (site_y-1) % dim_y,1], surface[(site_x-1) % dim_x, site_y % dim_y,1]]
    mid3_count = [mid3.count(metals[n]) for n in range(len(metals))]
    
    bot3 = [surface[(site_x-1) % dim_x, (site_y-1) % dim_y, 2], surface[(site_x-1) % dim_x, (site_y+1) % dim_y, 2], surface[(site_x+1) % dim_x, (site_y-1) % dim_y, 2]]
    bot3_count = [bot3.count(metals[n]) for n in range(len(metals))]
    
    return site1_count + top6_count + mid3_count + bot3_count

metals = ['Ag', 'Au', 'Cu', 'Pd', 'Pt']
three_metals_combinations = [] #List of possible combinations of the three
# Der skal være 35, ikke 125

for a in metals:
    for b in metals:
        for c in metals:
            three_metals_combinations.append(''.join(sorted([a, b, c])))
            
# Remove duplicates
three_metals_combinations = list(dict.fromkeys(three_metals_combinations)) # Let's encode it in a better way later

def hollow_site_vector(surface, site_x, site_y):
    
    # First encode the 3 neighbours
    blues = [surface[(site_x+1) % dim_x, site_y, 0], surface[site_x, (site_y+1) % dim_y, 0], surface[(site_x+1) % dim_x, (site_y+1) % dim_y, 0]]
    blues = "".join(sorted(blues))
    idx = three_metals_combinations.index(blues)
    blues = 35*[0]
    blues[idx] = 1
    
    # Then the next neighbours (green)
    greens = [surface[(site_x+2) % dim_x, site_y, 0], surface[site_x, (site_y+2) % dim_y, 0], surface[site_x, site_y, 0]]
    greens_count = [greens.count(metals[n]) for n in range(len(metals))]
    
    # Then the next neighbours (brown) # Kunne gøres smartere med list comprehension og to lister med +- zipped
    browns = [surface[(site_x + a) % dim_x, (site_y + b) % dim_y, c] for a, b, c in zip([1, 2, 2, 1, -1, -1], [2, 1, -1, -1, 1, 2], [0, 0, 0, 0, 0, 0])]
    browns_count = [browns.count(metals[n]) for n in range(len(metals))]
    
    # Then the three downstairs neighbours
    yellows = [surface[(site_x + a) % dim_x, (site_y + b) % dim_y, c] for a, b, c in zip([0, 1, 0], [0, 0, 1], [1, 1, 1])]
    yellows_count = [yellows.count(metals[n]) for n in range(len(metals))]
    
    # Then the purples downstairs
    purples = [surface[(site_x + a) % dim_x, (site_y + b) % dim_y, c] for a, b, c in zip([1, -1, 1], [-1, 1, 1], [1, 1, 1])]
    purples_count = [purples.count(metals[n]) for n in range(len(metals))]
    
    return blues + greens_count + browns_count + yellows_count + purples_count

def mixed_site_vector(surface, hol_site_x, hol_site_y, top_site_x, top_site_y):
    hol_site_vec = hollow_site_vector(surface, hol_site_x, hol_site_y)
    top_site_vec = on_top_site_vector(surface, top_site_x, top_site_y)
    mixed_site_vec = np.concatenate([hol_site_vec, top_site_vec])

    return mixed_site_vec

def pandas_to_DMatrix(df):#, label):
    label = pd.DataFrame(np.random.randint(2, size=len(df)))
    DMatrix = xgb.DMatrix(df)#, label=label)
    return DMatrix

#### FUNCTIONS FOR TRAINING MODELS ####

def learning_curve(model_name): #For regressor
    # retrieve performance metrics
    results = model_name.evals_result()
    epochs = len(results['validation_0']['mae'])
    x_axis = range(0, epochs)
    
    # plot log loss
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(x_axis, results['validation_0']['mae'], label='Train')
    ax.plot(x_axis, results['validation_1']['mae'], label='Validation')
    ax.legend()
    
    plt.xlabel("Epoch")
    plt.ylabel('Log Loss')
    plt.title('XGBoost Loss curve')
    plt.show()
    return None

def single_parity_plot(model_name, X_test, y_test_series, training_data, adsorbate):
    model_predictions = model_name.predict(X_test)
    
    model_type_title = "Gradient Boosting"
    #Fix sklearn LinearRegressions weird list of lists thing
    if len(np.shape(model_predictions)) == 2:
        #print("For søren, jeg har fået en LinearRegression model fra sklearn. Sikke skørt det er at returnere predictions som en liste af en liste. Det vil jeg straks rette op på")
        #print("model_predictions: ", model_predictions)
        model_predictions = model_predictions.reshape(-1)
        #print("model_predictions after reshaping: ", model_predictions)
        
        #Sørg for at den skriver linear regression model i titlen
        model_type_title = "Linear Regression"
    
    y_test = y_test_series.values.tolist()
    
    # Find MAE:
    errors = y_test_series.to_numpy().reshape(-1)-model_predictions
    MAE = np.mean(np.abs(errors))
    #print(f"MAE: {MAE:.3f}")

    if adsorbate == "H and O":
        #I want two plt.scatter, one for each adsorbate
        flat_list = [item for sublist in X_test[["adsorbate"]].values.tolist() for item in sublist]
        pred_H = [model_predictions[n] for n in range(len(y_test)) if flat_list[n] == 0]
        pred_O = [model_predictions[n] for n in range(len(y_test)) if flat_list[n] == 1]
        true_H = [y_test[n] for n in range(len(y_test)) if flat_list[n] == 0]
        true_O = [y_test[n] for n in range(len(y_test)) if flat_list[n] == 1]
        
        MAE_O = np.mean(np.abs(np.array(true_O).reshape(-1)-pred_O))
        MAE_H = np.mean(np.abs(np.array(true_H).reshape(-1)-pred_H))
        print(f"MAE(O): {MAE_O:.3f}")
        print(f"MAE(H): {MAE_H:.3f}")
    
    fig, ax1 = plt.subplots()
    
    if adsorbate == "H and O":
        ax1.scatter(true_H, pred_H, s = 20, c = "tab:green", label = "Adsorbate: H", marker = "$H$")
        ax1.scatter(true_O, pred_O, s = 20, c = "tab:red", label = "Adsorbate: O", marker = "$O$")
        
        ax1.set_title(model_type_title + " model predictions of $\Delta G_{*O}^{DFT} (eV)$ and $\Delta G_{*H}^{DFT} (eV)$ \n Training data: " + training_data)
    
    if adsorbate == "O":
        ax1.scatter(y_test_series, model_predictions, s = 20, c = "tab:red", label = "Adsorbate: O", marker = "$O$")
        ax1.set_title(model_type_title + " model predictions of $\Delta G_{*O}^{DFT} (eV)$ \n Training data: " + training_data)
        
    if adsorbate == "OH":
        ax1.scatter(y_test_series, model_predictions, s = 60, c = "tab:blue", label = "Adsorbate: OH", marker = "$OH$")
        ax1.set_title(model_type_title + " model predictions of $\Delta G_{*OH}^{DFT} (eV)$")
    
    if adsorbate == "H":
        ax1.scatter(y_test_series, model_predictions, s = 20, c = "tab:green", label = "Adsorbate: H", marker = "$H$")
        ax1.set_title(model_type_title + " model predictions of $\Delta G_{*H}^{DFT} (eV)$")

    if adsorbate == "COOH":
        ax1.scatter(y_test_series, model_predictions, s = 20, c = "cornflowerblue", label = "Adsorbate: COOH", marker = "x")
        ax1.set_title(model_type_title + " model predictions of $\Delta G_{*COOH}^{DFT} (eV)$")
    
    if adsorbate == "COOH+H":
        ax1.scatter(y_test_series, model_predictions, s = 20, c = "seagreen", label = "Adsorbate: COOH+H", marker = "x")
        ax1.set_title(model_type_title + " model predictions of $\Delta G_{*COOH+*H}^{DFT} (eV)$")

    
    if adsorbate == "CO":
        ax1.scatter(y_test_series, model_predictions, s = 60, c = "orangered", label = "Adsorbate: CO", marker = "$CO$")
        ax1.set_title(model_type_title + " model predictions of $\Delta G_{*CO}^{DFT} (eV)$")
    
    ax1.set_xlabel("$\Delta G_{*Adsorbate}^{DFT} (eV)$")
    ax1.set_ylabel("$\Delta G_{*Adsorbate}^{Pred} (eV)$")
    
    ax1.text(0.8, 2.4, f"MAE(test) = {MAE:.3f}", color="deepskyblue", fontweight='bold', fontsize = 12)
    
    left, bottom, width, height = [0.16, 0.65, 0.2, 0.2]
    ax_inset = fig.add_axes([left, bottom, width, height])
    
    pm, lw, fontsize = 0.1, 0.5, 14

    ax_inset.hist(errors, bins=np.arange(-0.6, 0.6, 0.05),
          color="deepskyblue",
          density=True,
          alpha=0.7,
          histtype='stepfilled',
          ec='black',
          lw=lw)
    
    # Make plus/minus 0.1 eV lines in inset axis
    ax_inset.axvline(pm, color='black', ls='--', dashes=(5, 5), lw=lw)
    ax_inset.axvline(-pm, color='black', ls='--', dashes=(5, 5), lw=lw)
    
    # Set x-tick label fontsize in inset axis
    ax_inset.tick_params(axis='x', which='major', labelsize=fontsize-6)
    
    # Remove y-ticks in inset axis
    ax_inset.tick_params(axis='y', which='major', left=False, labelleft=False)
    
    # Set x-tick locations in inset axis
    ax_inset.xaxis.set_major_locator(ticker.MultipleLocator(0.50))
    ax_inset.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    
    # Remove the all but the bottom spines of the inset axis
    for side in ['top', 'right', 'left']:
        ax_inset.spines[side].set_visible(False)
    
    # Make the background transparent in the inset axis
    ax_inset.patch.set_alpha(0.0)
    
    # Print 'pred-calc' below inset axis
    ax_inset.text(0.5, -0.33,
                  '$pred - DFT$ (eV)',
                  ha='center',
                  transform=ax_inset.transAxes,
                  fontsize=fontsize-7)
    
    # Make central and plus/minus 0.1 eV lines in scatter plot
    lims = [-0.3, 2.75]
    
    # Set x and y limits
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    
    ax1.plot(lims, lims,
            lw=lw, color='black', zorder=1,
            label=r'$\rm \Delta G_{pred} = \Delta G_{DFT}$')
    
    # Make plus/minus 0.1 eV lines around y = x
    ax1.plot(lims, [lims[0]+pm, lims[1]+pm],
            lw=lw, ls='--', dashes=(5, 5), color='black', zorder=1,
            label=r'$\rm \pm$ {:.1f} eV'.format(pm))
            
    ax1.plot([lims[0], lims[1]], [lims[0]-pm, lims[1]-pm],
            lw=lw, ls='--', dashes=(5, 5), color='black', zorder=1)
    
    ax1.legend(frameon=False,
          bbox_to_anchor=[0.45, 0.0],
          loc='lower left',
          handletextpad=0.2,
          handlelength=1.0,
          labelspacing=0.2,
          borderaxespad=0.1,
          markerscale=1.5,
          fontsize=fontsize-5)
    
    #plt.savefig(figure_folder + "Parity_trained_OH_tested_BOTH.png", dpi = 300, bbox_inches = "tight")
    # Save figure with a random name, rename later
    plt.savefig(figure_folder + str(time.time())[6:10]+str(time.time())[11:15], dpi = 300, bbox_inches = "tight")
    plt.show()
    return None

def prepare_csv(feature_folder, filename, adsorbate):
    init_df = pd.read_csv(feature_folder + filename)

    # Add a first column about the adsorbate
    adsorbate_df = pd.DataFrame([adsorbate for x in range(len(init_df))], columns = ["adsorbate"])

    #Combine
    prepared_df = pd.concat([adsorbate_df, init_df], axis = 1)
    return prepared_df

def return_mae(model_name, X_test, y_test_series): #Returns MAE on test set for a model (Either XGBoost or )
    model_predictions = model_name.predict(X_test)
    
    if len(np.shape(model_predictions)) == 2:
        model_predictions = model_predictions.reshape(-1)
    y_test = y_test_series.values.tolist()
    
    # Find MAE:
    errors = y_test_series.to_numpy().reshape(-1)-model_predictions
    MAE = np.mean(np.abs(errors))
    return MAE