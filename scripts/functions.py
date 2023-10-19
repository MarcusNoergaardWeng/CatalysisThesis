import sys
import xgboost;
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
import time
import random
import csv
from skopt import gp_minimize
from ase.db import connect

import scipy
import itertools as it

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

# Define Boltzmann's constant
kB = 1.380649e-4 / 1.602176634  # eV K-1 (exact)
kBT = kB*300 # eV

# Folder paths
log_folder = "../Coverage_logs/"

#### LAOADING AND PREPARING FEATURES DATA FROM CSV FILES ####

def prepare_dataset(feature_folder, filename):
    full_df = pd.read_csv(feature_folder + filename)

    all_cols = full_df.columns
    #Seperate the energies and remove the useless columns

    X = full_df.loc[:, :all_cols[-4]]
    y = full_df[["G_ads (eV)"]] #Der er åbenbart et mellemrum her, det forsvinder måske, hvis jeg laver features igen
    
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size = 0.2)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size = 0.5)
    return X_train, y_train, X_val, y_val, X_test, y_test

def combine_neighbour_features(feature_folder, db_name, filename_H, filename_COOH): #Tested, seems
    """This function combines the hollow vector and the on-top vector"""
    # Read the dataframes
    H_df = pd.read_csv(feature_folder + filename_H)
    COOH_df = pd.read_csv(feature_folder + filename_COOH)

    # Combine the dataframes
    combined_df = pd.merge(H_df, COOH_df, on = db_name + "row") #det hedder: db_name_SWR + "row"
    y = combined_df[["G_ads(eV)"]] #Do I need to hardcopy this in order to not pass by reference?

    # List of unwanted column names
    unwanted_columns = ['G_ads(eV)', 'G_ads (eV)', 'slab db row_y', 'slab db row_x', db_name + "row"]

    # Remove the unwanted columns from the combined dataframe
    X = combined_df.drop(columns=unwanted_columns)

    # Or just go straight to train, val, test - prob the good idea
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size = 0.2)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size = 0.5)
    return X_train, y_train, X_val, y_val, X_test, y_test

#### MAKING FEATURES FROM DFT DATA ####

def features_from_mixed_sites(db_folder, db_name, db_name_slab, features_folder, feature_file_H, feature_file_COOH):
    """Make features from slabs with mixed-sites (H + COOH as neighbours). 
    Two sets of features will be made. One with the feature vector based on H and one based on COOH.
    They will be saved seperately to be combined later
    Combine the feature vectors and save them to be loaded easily"""

    ### Code that makes H vectors (Given COOH as neighbour) ###

    # Initiate feature readers
    reader_COOH_H = FccStandard111(metals) #Hollow sites
    
    # Initiate counters of rejected samples
    rejected_COOH_H = 0

    # Writer headers to files
    with open(f'{features_folder}{feature_file_H}', 'w') as file_COOH_H:
        file_COOH_H.write(",".join([f"H_feature{n}" for n in range(55)]) + f',G_ads(eV),slab db row,{db_name}row')
    
    # Load HEA(111) databases
    with connect(f'{db_folder}{db_name}') as db_COOH_H,\
        connect(f'{db_folder}{db_name_slab}') as db_slab,\
        open(f'{features_folder}{feature_file_H}', 'a') as file_COOH_H:
        
        # Iterate through slabs without adsorbates
        for row_slab in db_slab.select('energy', H=0, C=0, O=0):
            # Iterate through the adsorbate
            for ads in ['H']:
                #print("A1")
                # Set adsorbate-specific parameters
                if ads == 'H':
                    db = db_COOH_H
                    kw = {'C':1, 'O': 2, 'H': 2} #Might need to change this to accomodate for the actual adsorbates
                    db_name = db_name
                    out_file = file_COOH_H
                    ads_atom = "H"
                    #print("A2")
                # Set counter of matched slabs between the databases to zero
                n_matched = 0

                # Get the corresponding slab with adsorbate
                for row in db.select('energy', **kw, **row_slab.count_atoms()): # It would be really nice if this line said something instead of just not running...
                    all_ads = "HCOOH"
                    if row.symbols[:-len(all_ads)] == row_slab.symbols: #Fix this line to accomodate for COOH+H
                        #print("A4")
                        # Increment the counter of matched structures
                        n_matched += 1

                        # Get atoms object
                        atoms = db.get_atoms(row.id)

                        # Make slab instance
                        slab = Slab(atoms, ads=ads, ads_atom='H')

                        # If the adsorbate is *H
                        if ads == 'H': # Der er meget mere kode i "Load_prep_DFT_data" under H, måske mangler jeg noget
                            
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
                                rejected_COOH_H += 1
                                continue

                            # Get features of structure
                            features = reader_COOH_H.get_features(slab, radius=2.6, site_ids=site_ids_H)
                        
                        # Get adsorption energy
                        E_ads = correct_DFT_energy_COOH_H(molecules_dict, row.energy, row_slab.energy) # This is the new formula

                        # Write output to file
                        features = ','.join(map(str, features))
                        out_file.write(f'\n{features},{E_ads:.6f},{row_slab.id},{row.id}')

                # Print a message if more than one slabs were matched. This probably means that
                # the same slab has accidentally been saved multiple to the database
                if n_matched > 1:
                    print(f'[INFO] {n_matched} {ads} and slab matched for row {row_slab.id} in') #{db_name_slab}')

        # Print the number of rejected samples to screen
        print('rejected COOH_H samples: ', rejected_COOH_H)

    ### Code that makes H vectors (Given H as neighbour) ###

    # Initiate feature readers
    reader_COOH_H = OntopStandard111(metals)

    # Initiate counters of rejected samples
    rejected_COOH_H = 0

    # Writer headers to files
    with open(f'{features_folder}{feature_file_COOH}', 'w') as file_COOH_H:
        column_names = [f"COOH_feature{n}" for n in range(20)]
        column_names.append('G_ads (eV)')
        column_names.append('slab db row')
        column_names.append(f'{db_name}row')
        file_COOH_H.write(",".join(column_names))

    # Load HEA(111) databases
    with connect(f'{db_folder}{db_name}') as db_COOH_H,\
        connect(f'{db_folder}{db_name_slab}') as db_slab,\
        open(f'{features_folder}{feature_file_COOH}', 'a') as file_COOH_H:

        # Iterate through slabs without adsorbates
        for row_slab in db_slab.select('energy', H=0, C=0, O=0):

            # Iterate through the two adsorbates
            for ads in ['COOH']:

                # Set adsorbate-specific parameters
                if ads == 'COOH':
                    db = db_COOH_H
                    kw = {'C':1, 'O': 2, 'H': 2}
                    db_name = db_name
                    out_file = file_COOH_H

                # Set counter of matched slabs between the databases to zero
                n_matched = 0

                # Get the corresponding slab with adsorbate
                for row in db.select('energy', **kw, **row_slab.count_atoms()):
                    #print(f"row.id: {row.id}")
                    
                    # If symbols match up
                    all_ads = "HCOOH"
                    if row.symbols[:-len(all_ads)] == row_slab.symbols:

                        # Increment the counter of matched structures
                        n_matched += 1

                        # Get atoms object
                        atoms = db.get_atoms(row.id) # Fjern H (atom 45) her
                        del atoms[45]
                        #new_atoms = Atoms(symbols=atoms.symbols[:-1], cell=atoms.cell)
                        # Make slab instance
                        slab = Slab(atoms, ads=ads, ads_atom='C')
                        
                        # If the adsorbate is *COOH
                        if ads == 'COOH':

                            # Get adsorption site elements as neighbors within a radius
                            site_elems, site = slab.get_adsorption_site(radius=2.6, hollow_radius=2.6)

                            # If the site does not consist of exactly one atom, then skip this sample
                            # as the *OH has moved too far away from an on-top site
                            try:
                                if len(site_elems) !=1:
                                    rejected_COOH_H += 1
                                    #slab.view()
                                    continue
                            except TypeError:
                                print(site_elems, site)
                                print(row_slab.id, row.id)
                                slab.view()
                                exit()

                            # Get features of structure
                            features = reader_COOH_H.get_features(slab, radius=2.6)

                        # Get adsorption energy
                        E_ads = correct_DFT_energy_COOH_H(molecules_dict, row.energy, row_slab.energy) # This is the new formula
                        #print(f"E_ads: {E_ads:.2f}")
                        
                        # Write output to file
                        features = ','.join(map(str, features))
                        out_file.write(f'\n{features},{E_ads:.6f},{row_slab.id},{row.id}')

                # Print a message if more than one slabs were matched. This probably means that
                # the same slab has accidentally been saved multiple to the database
                if n_matched > 1:
                    print(f'[INFO] {n_matched} {ads} and slab matched for row {row_slab.id} in') #{db_name_slab}')

    # Print the number of rejected samples to screen
    print('rejected COOH_H samples: ', rejected_COOH_H)
    return None

def features_from_DFT_data_H_and_COOH(features_folder, db_folder, db_name_H, db_name_COOH, db_name_slab, feature_file_H, feature_file_COOH):
    """Makes seperate features for H and COOH"""
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

def calc_correction_constant_H_COOH(AF):
    ### Summing up all the Approximation Factors for H+COOH
    ZpE_sum  = AF["bound_COOH"]["ZPE"]  + AF["bound_H"]["ZPE"]  - AF["HCOOH"]["ZPE"]  
    CpdT_sum = AF["bound_COOH"]["CpdT"] + AF["bound_H"]["CpdT"] - AF["HCOOH"]["CpdT"] 
    TS_sum   = AF["bound_COOH"]["TS"]   + AF["bound_H"]["TS"]   - AF["HCOOH"]["TS"]   #Check, that the signs are correct
    correction_constant_COOH_H = ZpE_sum + CpdT_sum - TS_sum
    return correction_constant_COOH_H

def correct_DFT_energy_COOH_H(molecules_dict, E_HplusCOOH, E_slab):
    DeltaE = E_HplusCOOH - molecules_dict["CH2O2"] - E_slab
    correction_constant_COOH_H = calc_correction_constant_H_COOH(AF)
    DeltaG = DeltaE + correction_constant_COOH_H
    #return DeltaG
    return DeltaE

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
    #return DeltaG
    return DeltaE

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
    #return DeltaG
    return DeltaE

def calc_correction_constant_O(AF):
    ### Summing up all the Approximation Factors for O
    ZpE_sum  = AF["bound_O"]["ZPE"]  - AF["H2O"]["ZPE"]  + AF["H2"]["ZPE"]
    CpdT_sum = AF["bound_O"]["CpdT"] - AF["H2O"]["CpdT"] + AF["H2"]["CpdT"]
    TS_sum   = AF["bound_O"]["TS"]   - AF["H2O"]["TS"]   + AF["H2"]["TS"]
    correction_constant_O = ZpE_sum + CpdT_sum - TS_sum
    return correction_constant_O

def calc_correction_constant_OH(AF):
    ### Summing up all the Approximation Factors for OH
    ZpE_sum  = AF["bound_OH"]["ZPE"]  - AF["H2O"]["ZPE"]  + 1/2*AF["H2"]["ZPE"]
    CpdT_sum = AF["bound_OH"]["CpdT"] - AF["H2O"]["CpdT"] + 1/2*AF["H2"]["CpdT"]
    TS_sum   = AF["bound_OH"]["TS"]   - AF["H2O"]["TS"]   + 1/2*AF["H2"]["TS"]
    correction_constant_OH = ZpE_sum + CpdT_sum - TS_sum
    return correction_constant_OH

#### MAKE SWIM RING SURFACE ####

def AB_to_split(A, B): #Tested, works
    split = [0, 0, 0, 0, 0]
    number = 2 #Set this to 2 for 1/3 and 6 for 1/7 swr surface
    if "Ag" in B:
        split[0] += number/len(B)
    if "Au" in B:
        split[1] += number/len(B)
    if "Cu" in B:
        split[2] += number/len(B)
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

def initialize_swim_surface(A, B, dim_x, dim_y): # Tested, works
    # Make the split from the metals used
    # A and B should be lists of the metals present in the swim rings and the insides
    # A is inside the swim rings and B is the swim rings
    # Soo this function doesn't ensure swim rings, it just makes a random mixture - Is what I thought, but the error was in the ase db software
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
    #surface = precompute_binding_energies_SPEED(surface, dim_x, dim_y, models)
    return surface

#### LOAD BINDING ENERGY MODELS ####

def load_E_models():
    """"This function loads the current best binding energy (G) prediction models trained on 
    both HEA (High-Entropy Alloy) and SWR (Swim-Ring) data.
    Except for CO, that I only have data on for the HEA slabs"""
    DeltaE_path = "../Models/DeltaE/"
    # Load H binding energy (G) prediction model - Trained on HEA and SWR data
    H_HEA_SWR_model = xgb.Booster({'nthread': 8})
    H_HEA_SWR_model.load_model(DeltaE_path+"H_HEA_SWR.model")

    # Load COOH binding energy (G) prediction model - Trained on HEA and SWR data
    COOH_HEA_SWR_model = xgb.Booster({'nthread': 8})
    COOH_HEA_SWR_model.load_model(DeltaE_path+"COOH_HEA_SWR.model")

    # Load mixed site energy (G)  prediction model - Trained on HEA and SWR data
    mixed_HEA_SWR_model = xgb.Booster({'nthread': 8})
    mixed_HEA_SWR_model.load_model(DeltaE_path+"COOH_H_HEA_SWR.model")

    ## Load models used only for the coverage simulations

    # Load OH binding energy (G) prediction model - Trained on HEA data
    OH_model = xgb.Booster({'nthread': 8})
    OH_model.load_model(DeltaE_path+"OH_HEA.model")

    # Load OH binding energy (G) prediction model - Trained on HEA data
    O_model = xgb.Booster({'nthread': 8})
    O_model.load_model(DeltaE_path+"O_HEA.model")

    ## Load OLD models only trained on HEA data

    # Load old H binding energy (G) model - Trained on HEA data only
    #H_HEA_model = xgb.Booster({'nthread': 8})
    #H_HEA_model.load_model(DeltaG_path+"H_HEA.model")

    # Load old COOH binding energy (G) model - Trained on HEA data only
    #COOH_HEA_model = xgb.Booster({'nthread': 8})
    #COOH_HEA_model.load_model(DeltaG_path+"COOH_HEA.model")

    # Load old H+COOH binding energy (G) model - Trained on HEA data only
    #mixed_HEA_model = xgb.Booster({'nthread': 8})
    #mixed_HEA_model.load_model(DeltaG_path+"COOH_H_HEA.model")

    models = {"H": H_HEA_SWR_model, "COOH": COOH_HEA_SWR_model, \
              "mixed": mixed_HEA_SWR_model, \
              "OH": OH_model, "O": O_model}
              #"H_old": H_HEA_model, "COOH_old": COOH_HEA_model, "mixed_old": mixed_HEA_model}
    return models

models = load_E_models() # I think I have to load it in here in order for the bayesian optimization scheme to work. I can't pass stuff to the functions

#### LOAD CORRECTIONS ####

def load_corrections():
    corrections = {
        "H_COOH": calc_correction_constant_H_COOH(AF), \
        "COOH": calc_correction_constant_COOH(AF), \
        "H": calc_correction_constant_H(AF), \
        "OH": calc_correction_constant_OH(AF), \
        "O": calc_correction_constant_O(AF), \
        "Bagger_H": 0.20, \
        "Bagger_COOH": -0.11414, \
        "Jack_H": 0.20,\
        "Jack_COOH": 0.29, \
        "Jack_H_Pt": 0.20 - 0.0676, \
        "Jack_COOH_Pt": 0.29 - 0.0676, \
        "Jack_H_bonus": 0.20 - 0.0676 - 0.04, \
        "Jack_COOH_bonus": 0.29 - 0.0676 - 0.04}
    return corrections

corrections = load_corrections()

#### LOAD PURE METAL ENERGIES ####

def load_SE_energies():
    DFT_folder = "../DFT_data/"
    # The COOH Pt data is in "single_element_COOH_C_adsorbed_out.db"
    db_name_SE_COOH = "single_element_COOH_C_adsorbed_out.db"
    # THe H data is in "single_element_H_out.db"
    db_name_SE_H = "single_element_H_out.db"
    db_name_SE = "single_element_slabs_out.db"

    SE_COOH_metals = []
    SE_COOH_energies = []

    with connect(f'{DFT_folder}{db_name_SE_COOH}') as db_COOH:
        for row_slab in db_COOH.select('energy'):
            SE_COOH_energies.append(row_slab.energy)
            SE_COOH_metals.append(row_slab.formula[0:2])

    SE_H_metals = []
    SE_H_energies = []

    with connect(f'{DFT_folder}{db_name_SE_H}') as db_H:
        for row_slab in db_H.select('energy'):
            SE_H_energies.append(row_slab.energy)
            SE_H_metals.append(row_slab.formula[0:2])
    
    SE_slab_metals = []
    SE_slab_energies = []

    with connect(f'{DFT_folder}{db_name_SE}') as db_slab:
        for row_slab in db_slab.select('energy'):
            SE_slab_energies.append(row_slab.energy)
            SE_slab_metals.append(row_slab.formula[0:2])

    DeltaE_COOH = np.array(SE_COOH_energies) - molecules_dict["CH2O2"] + 1/2*molecules_dict["H2"] - np.array(SE_slab_energies)
    DeltaE_H    = np.array(SE_H_energies) - np.array(SE_slab_energies) - 1/2*molecules_dict["H2"]

    pure_metal_info = {"DeltaE_H": DeltaE_H, "DeltaE_COOH": DeltaE_COOH, "SE_slab_metals": SE_slab_metals}
    return pure_metal_info

pure_metal_info = load_SE_energies()

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
    #plt.title('Number of good sites at each iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Estimated activity, excluding CO-poisoned sites')

    # Adjust the layout of the subplots
    plt.tight_layout()

    # Save figure
    #experiment_name = "140by140_100its_diagonal"
    #plt.savefig("../figures/DFT_calc_energies/"+experiment_name+"_progression.png", dpi = 300, bbox_inches = "tight")

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
            on_top_E = surface["COOH_E"][x_top][y_top]
            hollow_E = surface["H_E"][x_hollow][y_hollow]
            
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
def deltaEdeltaE_plot(filename, surface, title_text, pure_metal_info, reward_type, show_plot):
    
    # First, calculate the statistics of interest
    E_top_dict, E_hol_dict, good_hol_sites, n_ratios = sort_energies(surface, reward_type)

    fig, ax = plt.subplots(figsize = (6, 6))

    # Set the limits for both x and y axes
    xmin, xmax, ymin, ymax = -0.6, 1.3, -0.6, 1.3
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    # Set the major ticks and tick labels
    ax.set_xticks([-0.5, 0, 0.5, 1.0])
    ax.set_xticklabels([-0.5, 0, 0.5, 1.0])
    ax.set_yticks([-0.5, 0, 0.5, 1.0])
    ax.set_yticklabels([-0.5, 0, 0.5, 1.0])
    
    # Set the grid lines
    ax.grid(which='both', linestyle=':', linewidth=0.5, color='gray')
    
    #ax.set_title("Predicted energies for $^*COOH$ and $^*H$ for whole surface")
    ax.set_title(title_text)
    ax.set_xlabel("$\Delta E_{^*H}$ [eV]")
    ax.set_ylabel("$\Delta E_{^*COOH}$ [eV]")

    # Make lines at the correction constants
    ax.axhline(y = -corrections["Jack_COOH_bonus"], xmin = xmin, xmax = xmax, c = "black")
    ax.axvline(x = -corrections["Jack_H_bonus"], ymin = ymin, ymax = ymax, c = "black")

    # And text for those correction lines
    ax.text(x = -corrections["Jack_H_bonus"]+0.01, y =  1.2, s = "$\Delta E_{H_{UPD}, mod}$")
    ax.text(x =  0.3, y = -corrections["Jack_COOH_bonus"]+0.02, s = "$\Delta E_{FAOR, mod}$")

    #### REWARD TYPES ####

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
        ax.scatter(E_hol_dict[metal], E_top_dict[metal], label = f"{metal}$_{{{stochiometry[metal]:.1f}}}$", s = 0.5, alpha = 0.8, c = metal_colors[metal]) # edgecolor = "black", linewidth = 0.05, 
    
    for i, metal in enumerate(pure_metal_info["SE_slab_metals"]):
        ax.scatter(pure_metal_info["DeltaE_H"][i], pure_metal_info["DeltaE_COOH"][i], label = "Pure "+metal, marker = "o", c = metal_colors[metal], edgecolors='black')
        ax.text(pure_metal_info["DeltaE_H"][i]+0.03, pure_metal_info["DeltaE_COOH"][i], s = metal)

    ax.legend(loc="upper right")

    plt.savefig("../figures/"+filename+".png", dpi = 600, bbox_inches = "tight")
    if show_plot == True:
        plt.show()
    else:
        plt.close()
    return None

def deltaEdeltaE_plot_potential(filename, surface, potential, pure_metal_info, reward_type, show_plot):
    
    # First, calculate the statistics of interest
    E_top_dict, E_hol_dict, good_hol_sites, n_ratios = sort_energies(surface, reward_type)

    fig, ax = plt.subplots(figsize = (6, 6))

    # Set the limits for both x and y axes
    xmin, xmax, ymin, ymax = -0.6, 1.3, -0.6, 1.3
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    # Set the major ticks and tick labels
    ax.set_xticks([-0.5, 0, 0.5, 1.0])
    ax.set_xticklabels([-0.5, 0, 0.5, 1.0])
    ax.set_yticks([-0.5, 0, 0.5, 1.0])
    ax.set_yticklabels([-0.5, 0, 0.5, 1.0])
    
    # Set the grid lines
    ax.grid(which='both', linestyle=':', linewidth=0.5, color='gray')
    
    #ax.set_title("Predicted energies for $^*COOH$ and $^*H$ for whole surface")
    #ax.set_title(title_text)
    ax.set_xlabel("$\Delta E_{^*H}$ [eV]")
    ax.set_ylabel("$\Delta E_{^*COOH}$ [eV]")

    # Make lines at the correction constants
    ax.axhline(y = -corrections["Jack_COOH_bonus"], xmin = xmin, xmax = xmax, c = "black", linestyle = "dashed")
    ax.axvline(x = -corrections["Jack_H_bonus"], ymin = ymin, ymax = ymax, c = "black", linestyle = "dashed")

    # And text for those correction lines
    ax.text(x = -corrections["Jack_H_bonus"]-potential+0.01, y =  1.2, s = "$\Delta E_{H_{UPD, mod}}$")
    ax.text(x =  0.3, y = -corrections["Jack_COOH_bonus"]+0.02, s = "$\Delta E_{FAOR, mod}$")
    
    ### New dashed lines with potential
    # Lines that move with the deltaG=0 when potential (eU) changes
    ax.axhline(y = -corrections["Jack_COOH_bonus"]+potential, xmin = xmin, xmax = xmax, c = "black", linestyle = "dotted")
    ax.axvline(x = -corrections["Jack_H_bonus"]-potential, ymin = ymin, ymax = ymax,    c = "black", linestyle = "dotted")

    # Text showing the potential
    ax.text(x =  0.7, y = -corrections["Jack_COOH_bonus"]+potential+0.01, s = f"$eU = {potential:.2f}\,eV$") #\Delta E_{FAOR}$")

    stochiometry = surface["stochiometry"]
    for metal in ['Ag', 'Au', 'Cu', 'Pd', 'Pt']:
        ax.scatter(E_hol_dict[metal], E_top_dict[metal], label = f"{metal}$_{{{stochiometry[metal]:.2f}}}$", s = 0.5, alpha = 0.8, c = metal_colors[metal]) # edgecolor = "black", linewidth = 0.05, 
    
    for i, metal in enumerate(pure_metal_info["SE_slab_metals"]):
        ax.scatter(pure_metal_info["DeltaE_H"][i], pure_metal_info["DeltaE_COOH"][i], label = "Pure "+metal, marker = "o", c = metal_colors[metal], edgecolors='black')
        ax.text(pure_metal_info["DeltaE_H"][i]+0.03, pure_metal_info["DeltaE_COOH"][i], s = metal)

    ax.legend(loc="upper right")

    plt.savefig("../figures/"+filename+".png", dpi = 600, bbox_inches = "tight")
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
            on_top_E = surface["COOH_E"][x_top][y_top]
            hollow_E = surface["H_E"][x_hollow][y_hollow]
            
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
            on_top_E = surface["COOH_E"][x_top][y_top]
            hollow_E = surface["H_E"][x_hollow][y_hollow]
            
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
            on_top_E = surface["COOH_E"][x_top][y_top]
            hollow_E = surface["H_E"][x_hollow][y_hollow]
            
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
            on_top_E = surface["COOH_E"][x_top][y_top]
            hollow_E = surface["H_E"][x_hollow][y_hollow]
            
            # Find GOOD sites:
            if on_top_E < hollow_E: # The on-top binding energy is lower than hollow binding energy. Smaller means binds better
                # Here is a good site!
                n_under_diagonal += 1

    return -n_under_diagonal

#### FUNCTIONS FOR PREDITING ENERGIES ####

def calc_given_energies(surface):
    surface["COOH_given_H_down"]     = surface["mixed_down"]     - np.roll(surface["H_E"], (-1,  0), axis=(0, 1))
    surface["COOH_given_H_up_right"] = surface["mixed_up_right"] - np.roll(surface["H_E"], ( 0,  0), axis=(0, 1))
    surface["COOH_given_H_up_left"]  = surface["mixed_up_left"]  - np.roll(surface["H_E"], ( 0, -1), axis=(0, 1))

    surface["H_given_COOH_down"]     = surface["mixed_down"]     - np.roll(surface["COOH_E"], (+1,  0), axis=(0, 1))
    surface["H_given_COOH_up_right"] = surface["mixed_up_right"] - np.roll(surface["COOH_E"], ( 0,  0), axis=(0, 1))
    surface["H_given_COOH_up_left"]  = surface["mixed_up_left"]  - np.roll(surface["COOH_E"], ( 0, +1), axis=(0, 1))

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
    # HERHERHER LAV OM TIL E
    COOH_down_E = models["mixed"].predict(COOH_down_features_DM)
    COOH_up_right_E = models["mixed"].predict(COOH_up_right_features_DM)
    COOH_up_left_E = models["mixed"].predict(COOH_up_left_features_DM)

    # Make them into a nice matrix shape - in a minute
    COOH_down_E = np.reshape(COOH_down_E, (dim_x, dim_y))
    COOH_up_right_E = np.reshape(COOH_up_right_E, (dim_x, dim_y))
    COOH_up_left_E = np.reshape(COOH_up_left_E, (dim_x, dim_y))
    
    # Attach the energies to the matrices in the surface dictionary

    surface["mixed_down"]     = COOH_down_E
    surface["mixed_up_right"] = COOH_up_right_E
    surface["mixed_up_left"]  = COOH_up_left_E

    surface = calc_given_energies(surface)
    return surface

def initialize_surface(dim_x, dim_y, metals, split): #Is still random - could be used with a seed in the name of reproduceability
    dim_z = 3
    
    surf_atoms = create_surface(dim_x, dim_y, metals, split)
    
    # Adsorbates
    surf_ads_top = np.reshape(["empty"]*dim_x*dim_y, (dim_x, dim_y))
    surf_ads_hol = np.reshape(["empty"]*dim_x*dim_y, (dim_x, dim_y))

    # Binding energies
    surf_COOH_E = np.reshape([np.nan]*dim_x*dim_y, (dim_x, dim_y))# On-top sites
    surf_H_E    = np.reshape([np.nan]*dim_x*dim_y, (dim_x, dim_y))# Hollow sites

    # Mixed-site energies
    surf_COOH_down     = np.reshape([np.nan]*dim_x*dim_y, (dim_x, dim_y))
    surf_COOH_up_right = np.reshape([np.nan]*dim_x*dim_y, (dim_x, dim_y))
    surf_COOH_up_left  = np.reshape([np.nan]*dim_x*dim_y, (dim_x, dim_y))

    stochiometry = dict(zip(metals, np.array(split)/np.sum(split)))
    
    surf = {"atoms": surf_atoms, "stochiometry": stochiometry,\
            "ads_top": surf_ads_top, "ads_hol": surf_ads_hol, \
            "COOH_E": surf_COOH_E, "H_E": surf_H_E, "mixed_down": surf_COOH_down, "mixed_up_right": surf_COOH_up_right, "mixed_up_left": surf_COOH_up_left}
    return surf

def create_surface(dim_x, dim_y, metals, split):
    dim_z = 3
    num_atoms = dim_x*dim_y*dim_z
    if np.sum(split) != 1.0:
        # This split is not weighted properly, I'll fix it
        split = split / np.sum(split)
    #if split == "Even": #Had to remove, because of some stupid futurewarning
    #    proba = [1.0 / len(metals) for n in range(len(metals))] 
    #    surface = np.random.choice(metals, num_atoms, p=proba)
    
    surface = np.random.choice(metals, num_atoms, p=split)
    surface = np.reshape(surface, (dim_x, dim_y, dim_z))
    return surface

def precompute_binding_energies_TQDM(surface, dim_x, dim_y, models, predict_G_function): #TJEK I think this function can go faster if I make all the data first appended to a list, then to a PD and then 
    """A good example of how NOT to write code. Replaced by the rewritten _SPEED version"""
    for x, y in tqdm([(x, y) for x in range(dim_x) for y in range(dim_y)], desc = r"Predicting all ΔG", leave = False): # I could randomise this, so I go through all sites in a random order
        
        ads = "H"
        surface["H_E"][x][y] = predict_G_function(surface["atoms"], x, y, ads, models) ## A new function that wraps/uses the XGBoost model
        
        ads = "COOH"
        surface["COOH_E"][x][y] = predict_G_function(surface["atoms"], x, y, ads, models) ## A new function that wraps/uses the XGBoost model

    return surface

def predict_G(surface, site_x, site_y, adsorbate, models):
    """A good example of how NOT to write code. Replaced by the rewritten _SPEED version"""
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

def precompute_binding_energies_SPEED(surface, dim_x, dim_y, models): #TJEK ADD OH, H, mixed-site sådan at det kan bruges til coverage simulations
    hollow_features = [] # These features are the same for all hollow site adsorbates, hence these are made just once
    on_top_features = []

    # Make features for each site:
    for x, y in [(x, y) for x in range(dim_x) for y in range(dim_y)]:#, desc = r"Making all feature vectors", leave = True): # I could randomise this, so I go through all sites in a random order
        # Append the features for the hollow site adsorbates: H and O
        hollow_features.append([hollow_site_vector(surface["atoms"], x, y)])

        # Append the features for the on-top site adsorbates: OH and COOH
        on_top_features.append([on_top_site_vector(surface["atoms"], x, y)])

    # Remove the uneccesary singleton dimension
    hollow_features = np.squeeze(hollow_features)
    on_top_features = np.squeeze(on_top_features)

    # Make the features into a big dataframe
    hollow_features_df = pd.DataFrame(hollow_features, columns = [f"feature{n}" for n in range(55)])
    on_top_features_df = pd.DataFrame(on_top_features, columns = [f"feature{n}" for n in range(20)])

    # Turn them into DMatrix
    hollow_features_DM = pandas_to_DMatrix(hollow_features_df)
    on_top_features_DM = pandas_to_DMatrix(on_top_features_df)

    # Predict energies in one long straight line
    H_E    = models["H"].predict(hollow_features_DM) # HERE THE MODELS ARE CHOSEN
    O_E    = models["O"].predict(hollow_features_DM)
    COOH_E = models["COOH"].predict(on_top_features_DM)
    OH_E   = models["OH"].predict(on_top_features_DM)
    
    # Make them into a nice matrix shape - in a minute
    H_E    = np.reshape(H_E   , (dim_x, dim_y))
    O_E    = np.reshape(O_E   , (dim_x, dim_y))
    COOH_E = np.reshape(COOH_E, (dim_x, dim_y))
    OH_E   = np.reshape(OH_E   , (dim_x, dim_y))

    # Attach the energies to the matrices in the surface dictionary
    surface["H_E"]    = H_E
    surface["O_E"]    = H_E
    surface["COOH_E"] = COOH_E
    surface["OH_E"]   = OH_E

    # Add the thermal corrections to make Gibbs free energies
    surface["H_G"]    = surface["H_E"]    + corrections["Jack_H_bonus"]
    surface["COOH_G"] = surface["COOH_E"] + corrections["Jack_COOH_bonus"]
    surface["OH_G"]   = surface["OH_E"]   + corrections["OH"]
    surface["O_G"]    = surface["O_E"]    + corrections["O"]

    # Calculate and attach the border voltages
    surface["H_V"]    = calc_V_border(ads = "H",    G = surface["H_G"]   ) # TJek - should be based on G's I think
    surface["O_V"]    = calc_V_border(ads = "O",    G = surface["O_G"]   )
    surface["COOH_V"] = calc_V_border(ads = "COOH", G = surface["COOH_G"])
    surface["OH_V"]   = calc_V_border(ads = "OH",   G = surface["OH_G"]  )

    # Predict the energies on the mixed sites
    surface = predict_mixed_energies(surface, dim_x, dim_y, models)

    # Calculate the "*COOH given *H" and "*H given *COOH" energies
    surface = calc_given_energies(surface)

    return surface

def precompute_binding_energies_SPEED2(surface, dim_x, dim_y, models): #TJEK ADD OH, H, mixed-site sådan at det kan bruges til coverage simulations
    hollow_features = [] # These features are the same for all hollow site adsorbates, hence these are made just once
    on_top_features = []

    # Make features for each site:
    for x, y in [(x, y) for x in range(dim_x) for y in range(dim_y)]:#, desc = r"Making all feature vectors", leave = True): # I could randomise this, so I go through all sites in a random order
        # Append the features for the hollow site adsorbates: H and O
        hollow_features.append([hollow_site_vector(surface["atoms"], x, y)])

        # Append the features for the on-top site adsorbates: OH and COOH
        on_top_features.append([on_top_site_vector(surface["atoms"], x, y)])

    # Remove the uneccesary singleton dimension
    hollow_features = np.squeeze(hollow_features)
    on_top_features = np.squeeze(on_top_features)

    # Make the features into a big dataframe
    hollow_features_df = pd.DataFrame(hollow_features, columns = [f"feature{n}" for n in range(55)])
    on_top_features_df = pd.DataFrame(on_top_features, columns = [f"feature{n}" for n in range(20)])

    # Turn them into DMatrix
    hollow_features_DM = pandas_to_DMatrix(hollow_features_df)
    on_top_features_DM = pandas_to_DMatrix(on_top_features_df)

    # Predict energies in one long straight line
    H_E    = models["H"].predict(hollow_features_DM) # HERE THE MODELS ARE CHOSEN
    #O_E    = models["O"].predict(hollow_features_DM)
    COOH_E = models["COOH"].predict(on_top_features_DM)
    #OH_E   = models["OH"].predict(on_top_features_DM)
    
    # Make them into a nice matrix shape - in a minute
    H_E    = np.reshape(H_E   , (dim_x, dim_y))
    #O_E    = np.reshape(O_E   , (dim_x, dim_y))
    COOH_E = np.reshape(COOH_E, (dim_x, dim_y))
    #OH_E   = np.reshape(OH_E   , (dim_x, dim_y))

    # Attach the energies to the matrices in the surface dictionary
    surface["H_E"]    = H_E
    #surface["O_E"]    = H_E
    surface["COOH_E"] = COOH_E
    #surface["OH_E"]   = OH_E

    # Add the thermal corrections to make Gibbs free energies
    surface["H_G"]    = surface["H_E"]    + corrections["Jack_H_bonus"]
    surface["COOH_G"] = surface["COOH_E"] + corrections["Jack_COOH_bonus"]

    # Calculate and attach the border voltages
    #surface["H_V"]    = calc_V_border(ads = "H",    G = surface["H_E"]   ) # TJek - should be based on G's I think
    #surface["O_V"]    = calc_V_border(ads = "O",    G = surface["O_E"]   )
    #surface["COOH_V"] = calc_V_border(ads = "COOH", G = surface["COOH_E"])
    #surface["OH_V"]   = calc_V_border(ads = "OH",   G = surface["OH_E"]  )

    # Predict the energies on the mixed sites
    #surface = predict_mixed_energies(surface, dim_x, dim_y, models)

    # Calculate the "*COOH given *H" and "*H given *COOH" energies
    #surface = calc_given_energies(surface)

    return surface

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

def train_XGB_model(model_name, adsorbate, X_train, y_train, X_val, y_val, X_test, y_test):
    # Prepare XGBoost
    eval_set = [(X_train, y_train), (X_val, y_val)]
    XGBModel =  XGBRegressor(learning_rate = 0.1 #learning rate
                                , max_depth = 5     #maximum tree depth
                                , n_estimators = 500 #number of boosting rounds
                                , n_jobs = 8 #number of threads
                                , use_label_encoder = False)

    XGBModel.fit(X_train, y_train
                       , eval_set = eval_set
                       , early_stopping_rounds = 5
                       , eval_metric = ["mae"]
                       , verbose = False) #evals
    
    # Save model in the /models folder
    XGBModel.save_model("../models/" + model_name + ".model")

    learning_curve(XGBModel, model_name)

    score = XGBModel.score(X_train, y_train) #Det må man ikke før den er blevet trænet
    print("Training score: ", score)

    score = XGBModel.score(X_val, y_val) #Det må man ikke før den er blevet trænet
    print("Validation score: ", score)

    score = XGBModel.score(X_test, y_test) #Det må man ikke før den er blevet trænet
    print("Test score: ", score)

    single_parity_plot(XGBModel, model_name, X_test, y_test, adsorbate, adsorbate)

    return None

def learning_curve(model, model_name): #For regressor
    # retrieve performance metrics
    results = model.evals_result()
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
    figure_folder = "../figures/Loss_curves/DeltaE/"
    plt.savefig(figure_folder + model_name + "_Loss_Curve", dpi = 300, bbox_inches = "tight")
    plt.show()
    return None

def single_parity_plot(model, model_name, X_test, y_test_series, training_data, adsorbate):
    model_predictions = model.predict(X_test)
    
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
        ax1.set_title(model_type_title + " model predictions of $\Delta E_{*O}^{DFT} (eV)$")
        
    if adsorbate == "OH":
        ax1.scatter(y_test_series, model_predictions, s = 60, c = "tab:blue", label = "Adsorbate: OH", marker = "$OH$")
        ax1.set_title(model_type_title + " model predictions of $\Delta E_{*OH}^{DFT} (eV)$")
    
    if adsorbate == "H":
        ax1.scatter(y_test_series, model_predictions, s = 20, c = "tab:green", label = "Adsorbate: H", marker = "$H$")
        ax1.set_title(model_type_title + " model predictions of $\Delta E_{*H}^{DFT} (eV)$")

    if adsorbate == "COOH":
        ax1.scatter(y_test_series, model_predictions, s = 20, c = "cornflowerblue", label = "Adsorbate: COOH", marker = "x")
        ax1.set_title(model_type_title + " model predictions of $\Delta E_{*COOH}^{DFT} (eV)$")
    
    if adsorbate == "COOH+H":
        ax1.scatter(y_test_series, model_predictions, s = 20, c = "seagreen", label = "Adsorbate: COOH+H", marker = "x")
        ax1.set_title(model_type_title + " model predictions of $\Delta E_{*COOH+*H}^{DFT} (eV)$")

    
    if adsorbate == "CO":
        ax1.scatter(y_test_series, model_predictions, s = 60, c = "orangered", label = "Adsorbate: CO", marker = "$CO$")
        ax1.set_title(model_type_title + " model predictions of $\Delta G_{*CO}^{DFT} (eV)$")
    
    ax1.set_xlabel("$\Delta E_{*Adsorbate}^{DFT} (eV)$")
    ax1.set_ylabel("$\Delta E_{*Adsorbate}^{Pred} (eV)$")
    
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
            label=r'$\rm \Delta E_{pred} = \Delta E_{DFT}$')
    
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
    figure_folder = "../figures/DeltaG_models/DeltaE/"
    plt.savefig(figure_folder + model_name, dpi = 300, bbox_inches = "tight")
    #plt.savefig(figure_folder + str(time.time())[6:10]+str(time.time())[11:15], dpi = 300, bbox_inches = "tight")
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

#### RUNNING COVERAGE SIMULATIONS BASED ON SURFACE AND G PREDICTION MODELS ####

#def initialize_surface_coverage_simulation(dim_x, dim_y, metals, split): #Is still random - could be used with a seed in the name of reproduceability
#    dim_z = 3
#    #surface_list = np.array([int(dim_x*dim_y*dim_z/len(metals))*[metals[metal_number]] for metal_number in range(len(metals))]).flatten() #Jack had a way shorter way of doing this, but I think it was random drawing instead of ensuring a perfectly even split
#    #np.random.shuffle(surface_list) #Shuffle list
#    #surf_atoms = np.reshape(surface_list, (dim_x, dim_y, dim_z)) #Reshape list to the
#    
#    surf_atoms = create_surface(dim_x, dim_y, metals, split)
#    
#    # Adsorbates
#    surf_ads_top = np.reshape(["empty"]*dim_x*dim_y, (dim_x, dim_y))
#    surf_ads_hol = np.reshape(["empty"]*dim_x*dim_y, (dim_x, dim_y))
#    
#    # Binding energies
#    surf_COOH_G = np.reshape([np.nan]*dim_x*dim_y, (dim_x, dim_y)) # On-top sites
#    surf_H_G    = np.reshape([np.nan]*dim_x*dim_y, (dim_x, dim_y)) # Hollow sites
#    surf_OH_G    = np.reshape([np.nan]*dim_x*dim_y, (dim_x, dim_y))# On-top sites
#    surf_O_G    = np.reshape([np.nan]*dim_x*dim_y, (dim_x, dim_y)) # Hollow sites
#    
#    # Ad/desorbs at voltage (At which voltage is the binding energy 0?)
#    surf_COOH_V = np.reshape([np.nan]*dim_x*dim_y, (dim_x, dim_y)) # On-top sites
#    surf_H_V    = np.reshape([np.nan]*dim_x*dim_y, (dim_x, dim_y)) # Hollow sites
#    surf_OH_V    = np.reshape([np.nan]*dim_x*dim_y, (dim_x, dim_y))# On-top sites
#    surf_O_V    = np.reshape([np.nan]*dim_x*dim_y, (dim_x, dim_y)) # Hollow sites
#    
#    surf = {"atoms": surf_atoms, "ads_top": surf_ads_top, "ads_hol": surf_ads_hol, \
#            "COOH_G": surf_COOH_G, "H_G": surf_H_G, "OH_G": surf_OH_G, \
#            "O_G": surf_O_G, "COOH_V": surf_COOH_V, "H_V": surf_H_V, \
#            "OH_V": surf_OH_V, "O_V": surf_O_V} #This line had a but that took me two days to find... Pass by reference is such a smart feature (y)
#    return surf

def voltage_sweep(start, end, scan_rate):
    return np.linspace(start, end, int(np.abs(start - end) / scan_rate))

def calc_V_border(ads, G): # TJEK use this in an efficient way in the precompute speed function. Should take in a matrix and do the operation as a matrix calculation thing.
    """This function returns the border-voltage, at which the adsorbate adsorbs or desorbs
    I think I can just pass whole numpy arrays through this"""
    
    if ads == "H":
     V_border = - G  # Lille boost #+ 0.7 # HER sætter jeg lige et boost ind, for at tjekke, om det fungerer, når *CO-reaktionen sker
    if ads == "COOH":
        V_border = G -0.3 # TJEK er 0.7 et boost for at tjekke H+COOH -> CO + H2O reaktionen?
    if ads == "OH":
        V_border = G + 0.5 # TJEK BOOST            # Hvad er funktionen? Det samme som for COOH?
    if ads == "O":
        V_border = G/2+ 0.5 # TJEK BOOST          # Hvad er funktionen? Her hopper 2 elektroner af, så der sker noget andet
    
    return V_border

def create_log_file(file_name, column_names):
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)
    return None

def append_to_log_file(file_name, data):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)
    return None

def look_at_all_sites_and_adsorbates(simulation_name, surface, dim_x, dim_y, site_types, voltage, start_time):
    
    ## Look through all sites (not bridge sites yet) by index
    for site_x, site_y in [(x, y) for x in range(dim_x) for y in range(dim_y)]:
        
        ## Look through all adsorbates and their respective adsorption sites
        for ads, site_type in [(ads, site_types[ads]) for ads in ["H", "COOH", "OH", "O"]]: 
            #print(site_x, site_y, ads, site_type)
            
            # Make the decision to adsorb/desorb or do nothing with this function:
            surface = decision_to_leave(simulation_name, surface, site_x, site_y, ads, site_type, voltage, start_time)
    
    return surface

def decision_to_leave(simulation_name, surface, site_x, site_y, ads, site_type, voltage, start_time):
    ## Figure out if anything should be ad or desorbed
    
    # Is the adsorbate sitting at the site already?
    contents_of_site = surface[site_type][site_x][site_y]
    #is_ads_there = contents_of_site != "empty"
    
    # Should the adsorbate be there? #tested, seems to work
    V_border = surface[ads+"_V"][site_x][site_y]
    
    # H adsorbs when the voltage is BELOW the border voltage
    if ads == "H":
        supposed_to = voltage < V_border
    
    # COOH adsorbs when the voltage is ABOVE the border voltage
    if ads in ["COOH", "OH", "O"]:
        supposed_to = voltage > V_border
    
    # IFF the site is empty AND the adsorbate is supposed to be there, ADSORB
    if (contents_of_site == "empty") and (supposed_to == True): #Should check if the adsorbate there is == the adsorbate we're looking at
        # Adsorb
        surface[site_type][site_x][site_y] = ads
        
        # Save line to log file
        append_to_log_file(log_folder + simulation_name + ".csv", [ads, "adsorb", surface[ads+"_G"][site_x, site_y], surface[ads+"_V"][site_x, site_y], voltage, site_x, site_y, site_type, time.time() - start_time])
    
    # IFF the adsorbate is sitting at the site, and it shouldn't be, DESORB
    if (contents_of_site == ads) and (supposed_to == False): #ahh, this line made OH try to remove a COOH
        # Desorb
        surface[site_type][site_x][site_y] = "empty"
        
        # Save line to log file
        append_to_log_file(log_folder + simulation_name + ".csv", [ads, "desorption", surface[ads+"_G"][site_x, site_y], surface[ads+"_V"][site_x, site_y], voltage, site_x, site_y, site_type, time.time() - start_time])
    return surface

def shuffle(liste):
    #Shuffle the list
    random.shuffle(liste)
    return liste

def neighbours(surface, x, y, x_diff, y_diff):
    Top_site = surface["ads_top"][x][y]
    FCC_neighbour = surface["ads_hol"][(x + x_diff) % dim_x][(y + y_diff) % dim_y]
    
    if Top_site == "COOH" and FCC_neighbour == "H":
        return True
    else:
        return False
    
def reaction_CO(surface, x, y, x_diff, y_diff, voltage, start_time, simulation_name): #So I need another 2 functions like this: CO + O -> CO2 and CO + OH -> CO2 + 
    # Remove the *COOH
    surface["ads_top"][x][y] = "empty"
    append_to_log_file(log_folder + simulation_name + ".csv", ["COOH", "Make CO reaction", surface["COOH_G"][x, y], surface["COOH_V"][x, y], voltage, x, y, "ads_top", time.time() - start_time])
    
    # Remove the *H
    surface["ads_hol"][x+x_diff][y+y_diff] = "empty"
    append_to_log_file(log_folder + simulation_name + ".csv", ["H", "Make CO reaction", surface["H_G"][x, y], surface["H_V"][x, y], voltage, x, y, "ads_hol", time.time() - start_time])
     
    # Put a *CO instead of *COOH
    surface["ads_top"][x][y] = "CO"
    append_to_log_file(log_folder + simulation_name + ".csv", ["CO", "Make CO reaction", "n/a", "n/a", voltage, x+x_diff, y+y_diff, "ads_top", time.time() - start_time])
    
    return surface

def decision_to_react_CO(surface, voltage, start_time, simulation_name):

    # Look through all on-top sites for COOH species:
    for x, y in shuffle([(x, y) for x in range(dim_x) for y in range(dim_y)]): # Mixed order
        for x_diff, y_diff in shuffle([(0, 0), (0, -1), (-1, 0)]):             # Mixed order

            if neighbours(surface, x, y, x_diff, y_diff): #Are there H + COOH neighbours?

                surface = reaction_CO(surface, x, y, x_diff, y_diff, voltage, start_time, simulation_name)
                
                # Tjek - er det mon her de to/tre nye reaktioner skal tilføjes?
    return surface     

def decision_to_react_O(surface, voltage, start_time, simulation_name):
    # Look through all on-top sites for CO species:
    for x, y in shuffle([(x, y) for x in range(dim_x) for y in range(dim_y)]): # Mixed order
        for x_diff, y_diff in shuffle([(1, -1), (-1, -1), (-1, 1)]):           # Mixed order
            
            # If we find CO at the top-site x, y AND O at the x+x_diff, y+y_diff hollow_site: Remove both
            if (surface["ads_top"][x][y] == "CO") and (surface["ads_hol"][(x+x_diff) % dim_x][(y+y_diff) % dim_y] == "O"):
                
                #REACTION HAPPENS
                # REMOVE BOTH ADSORBATES AND WRITE LOGS
                # Remove the *CO
                surface["ads_top"][x][y] = "empty"
                append_to_log_file(log_folder + simulation_name + ".csv", ["CO", "CO oxidation", "n/a", "n/a", voltage, x, y, "ads_top", time.time() - start_time])
                
                # Remove the *O
                surface["ads_hol"][(x+x_diff) % dim_x][(y+y_diff) % dim_y] = "empty"
                append_to_log_file(log_folder + simulation_name + ".csv", ["O", "CO oxidation", "n/a", "n/a", voltage, (x+x_diff) % dim_x, (y+y_diff) % dim_y, "ads_hol", time.time() - start_time])
     
    return surface

#### FUNCTIONS FOR STATISTICS AND DATA VISUALIZATION FOR COVERAGE SIMULATIONS ####

def count_pairs(surface):
    '''This founction looks through a surface and returns the number of COOH + H pairs.
    It loops through all on-top sites, if there is a COOH, it looks at the neighbouring FCC sites
    If there'''
    pairs = 0
    # Look through all on-top sites for COOH species:
    for x, y in [(x, y) for x in range(dim_x) for y in range(dim_y)]:
        # Is there a COOH?
        if surface["ads_top"][x][y] == "COOH":
            # Are there any H on the neighbouring positions?
            for x_diff, y_diff in [(0, 0), (0, -1), (-1, 0)]:
                FCC_neighbour = surface["ads_hol"][(x + x_diff) % dim_x][(y + y_diff) % dim_y]
                if FCC_neighbour == "H":
                    pairs += 1          
    return pairs

def count_statistics(surface, voltage, statistics_log): # Should I input and return the statistics log each time or just append to it from inside the function? That's probably the way.
    '''This function counts some important metrics on the surface'''
    ## Log the voltage
    statistics_log["voltages"].append(voltage)
    
    ## Count number of the different adsorbates:
    # Count H adsorbates
    statistics_log["n_H"].append(np.count_nonzero(surface["ads_hol"] == 'H'))
    
    # Count COOH adsorbates
    statistics_log["n_COOH"].append(np.count_nonzero(surface["ads_top"] == 'COOH'))
    
    # Count COOH adsorbates
    statistics_log["n_OH"].append(np.count_nonzero(surface["ads_top"] == 'OH'))
    
    # Count COOH adsorbates
    statistics_log["n_O"].append(np.count_nonzero(surface["ads_hol"] == 'O'))
    
    # Count CO adsorbates
    statistics_log["n_CO"].append(np.count_nonzero(surface["ads_top"] == 'CO'))
    
    # Count H + COOH pairs
    statistics_log["n_pairs"].append(count_pairs(surface))
    
    return statistics_log

def initialize_statistics_log():
    statistics_log = {"voltages": [], "n_H": [], "n_COOH": [], "n_OH": [], "n_O": [], "n_CO": [], "n_pairs": []}
    return statistics_log

def plot_statistics_log(statistics_log, simulation_name, mode):
    fig, ax1 = plt.subplots()
    plt.title('Coverage Simulation (H+COOH) (Simple)')
    
    # Plot data on ax1
    line1 = ax1.plot(statistics_log["voltages"], statistics_log["n_H"], label="$^*H$", c="royalblue")
    line2 = ax1.plot(statistics_log["voltages"], statistics_log["n_COOH"], label="$^*COOH$", c="cornflowerblue")
    line3 = ax1.plot(statistics_log["voltages"], statistics_log["n_OH"], label="$^*OH$", c="orangered")
    line4 = ax1.plot(statistics_log["voltages"], statistics_log["n_O"], label="$^*O$", c="red")
    line5 = ax1.plot(statistics_log["voltages"], statistics_log["n_CO"], label="$^*CO$", c="black")
    
    ax1.set_xlabel('Voltage')
    ax1.set_ylabel('Number of adsorbates', color='blue')
    ax1.tick_params('y', colors='blue')
    
    #ax2 = ax1.twinx()
    
    #if mode == "pairs":
    #    # Plot data on ax2
    #    line5 = ax2.plot(statistics_log["voltages"], statistics_log["n_pairs"], label="$^*H$+$^*COOH$ pairs", c="tomato")
    #    ax2.set_ylabel('Number of H+COOH pairs', color='tab:red')
    #    lines = line1 + line2 + line3 + line4 + line5 
    #    
    #if mode == "CO":
    #    line5 = ax2.plot(statistics_log["voltages"], statistics_log["n_CO"], label="$^*CO$", c="tomato")
    #    ax2.set_ylabel('Number of $^*CO$ adsorbates', color='tab:red')
    #    lines = line1 + line2 + line3 + line4 + line5 
    #    
    #if mode == "plain":
    #    #do nothing (y)
    #    lines = line1 + line2 + line3 + line4
    #ax2.tick_params('y', colors='tab:red')
    
    # Combine the lines for the legend
    lines = line1 + line2 + line3 + line4 + line5 
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.savefig("../figures/coverage_simulations/"+simulation_name+".png", dpi=300, bbox_inches="tight")
    plt.show()
    return None

def plot_surface_composition(metals, surface):
    ### SURFACE COMPOSITION ###
    fig, ax_a = plt.subplots(1, 1, figsize = (6.4, 4.8))
    composition, composition_string = surface_composition(metals, surface)
    ymax = max(composition.values()) * 1.1
    ax_a.set_ylim(0, ymax)
    ax_a.set_title("Surface composition: " + composition_string)
    ax_a.set_yticks([])
    #ax.set_xticks([])
    bar1 = ax_a.bar(composition.keys(), composition.values(), alpha = 0.9, color = ["silver", "gold", "darkorange", "lightsteelblue", "cornflowerblue"])
    
    for idx, rect in enumerate(bar1):
        height = rect.get_height()
        fraction = list(composition.values())[idx]*100
        ax_a.text(rect.get_x() + rect.get_width() / 2.0, height, s = f"{fraction:.2f}" + " %", ha='center', va='bottom')
    return None

def surface_composition(metals, surface):
    composition = {}
    num_atoms = np.shape(surface)[0]*np.shape(surface)[1]*np.shape(surface)[2]
    
    for metal in metals:
        composition[metal] = np.count_nonzero(surface == metal) / num_atoms
        
    # Lav en kemisk formel for overfladen og lav subskript med deres fractions
    composition_string = ""

    for metal in metals:
        if composition[metal] > 0:
            composition_string += str(metal) + f"$_{{{composition[metal]:.2f}}}$"
    return composition, composition_string

#### Estimating Activities ####



def per_site_activity_special(COOH_binding_energies, H_binding_energies, eU):
    """This activity estimation function returns the 
    activity per surface atom.
    The activity is estimated based on the COOH binding
    energies (DeltaG), but the COOH energies from sites, 
    where H binds to a neighbour hollow site are not included
    Hence, they will contribute with 0 activity"""

    allowed_energies = [] #All the COOH binding energies on sites, where H is not a neighbouring adsorbate
    disqualified_energies = 0

    for idx_x in range(dim_x):
        for idx_y in range(dim_y):
            # Check if any of the neighbouring hollow sites could house an H
            security_clearence = True
            for x_diff, y_diff in [(0, 0), (-1, 0), (0, -1)]:
                neighbour_H_G = H_binding_energies[idx_x+x_diff, idx_y+y_diff]
                if neighbour_H_G < 0:
                    disqualified_energies += 1
                    security_clearence = False
                    continue
            
            if security_clearence == True:
                # If we get here it means no H + COOH disprop reac
                allowed_energies.append(COOH_binding_energies[idx_x][idx_y])
    
    j_avg = per_site_activity(allowed_energies, eU, jD=1.)
    return j_avg

def per_site_activity(energies, eU, jD=1.):
    """This activity estimation function returns the 
    activity per surface atom.
    The activity is estimated based on the COOH binding
    energies (DeltaG), but the COOH energies from sites, 
    where H binds to a neighbour hollow site
    The highest possible activity is 0.5 because of jD=1.
    Calculated using Angewandte Chemie equations 2-4 
    (doi: 10.1002/anie.202014374). Based on a function 
    Jack wrote in https://seafile.erda.dk/seafile/d/8586692f13/files/?p=%2Fscripts%2F__init__.py"""

    E_opt = -0.17
    energies = np.array(energies)
    energies = energies.flatten()
    n_surface_atoms = dim_x*dim_y

    # Making a list of activities

    # I used this initially, but as Jack said, 
    #jki = np.exp((-np.abs(energies - E_opt) -0.17 - eU) / kBT)

    G_RLS = (np.abs(energies - E_opt) +0.17 - eU) / kBT
    jki = np.exp(-G_RLS)
    j_avg = np.sum(1. / (1. / jki + 1./jD)) / n_surface_atoms
    return j_avg

## I need a function, where I just input a stoichiometry and a voltage, and 
# Create surface
# Loop through potentials
# surface = regn deltaG'er ud på ny og inkludér potentialet
# Kør aktivitetsudregnerne

def activity_of_surface(stoichiometry, V_min=-0.15, V_max=0.2, SPEED = "Bayesian"):
    # Create surface
    HEA_surface = initialize_surface(dim_x, dim_y, metals, stoichiometry)
    HEA_surface = precompute_binding_energies_SPEED2(HEA_surface, dim_x, dim_y, models)

    # Loop through potentials
    potential_range = np.linspace(V_min, V_max, 30) #HERHERHERHERHER
    j_avg_list         = []
    j_avg_special_list = []
    for eU in potential_range:

        # Calculate binding energies based on the potentials also
        COOH_binding_energies = HEA_surface["COOH_G"] - eU
        H_binding_energies    = HEA_surface["H_G"]    + eU

        # Use per_site_activity
        if SPEED == False: 
            j_avg_list.append(per_site_activity(COOH_binding_energies, eU, jD=1.))

        # Use per_site_activity_special
        j_avg_special_list.append(per_site_activity_special(COOH_binding_energies, H_binding_energies, eU))
    
    # Find the highest activity
    # Find the index of the maximum value in j_avg_special_list
    max_index = j_avg_special_list.index(max(j_avg_special_list))

    # Get the corresponding potential from potential_range
    max_potential = potential_range[max_index]
    if SPEED == False:
        activity_dict = {"potential_range": potential_range, "j_avg_list": j_avg_list, "j_avg_special_list": j_avg_special_list, "special_max_j": max(j_avg_special_list), "special_max_eU": max_potential, "stoichiometry": stoichiometry}
        return activity_dict
    if SPEED == True:
        activity_dict = {"j_avg_special_list": j_avg_special_list, "special_max_j": max(j_avg_special_list), "special_max_eU": max_potential}
        return activity_dict
    if SPEED == "Bayesian":
        return -max(j_avg_special_list)
    
def activity_directly_from_surface(surface, n_points, V_min=-0.15, V_max=0.2):
    # Loop through potentials
    potential_range = np.linspace(V_min, V_max, n_points) #HERHERHERHERHER
    j_avg_list         = []
    j_avg_special_list = []
    for eU in potential_range:
        # Calculate binding energies based on the potentials also
        COOH_binding_energies = surface["COOH_G"] - eU
        H_binding_energies    = surface["H_G"]    + eU

        # Use per_site_activity 
        j_avg_list.append(per_site_activity(COOH_binding_energies, eU, jD=1.))

        # Use per_site_activity_special
        j_avg_special_list.append(per_site_activity_special(COOH_binding_energies, H_binding_energies, eU))
    
    # Find the highest activity
    # Find the index of the maximum value in j_avg_special_list
    max_index = j_avg_special_list.index(max(j_avg_special_list))

    # Get the corresponding potential from potential_range
    max_potential = potential_range[max_index]

    activity_dict = {"potential_range": potential_range, "j_avg_list": j_avg_list, "j_avg_special_list": j_avg_special_list, "special_max_j": max(j_avg_special_list), "special_max_eU": max_potential, "stoichiometry": surface["stochiometry"]}
    return activity_dict

def stoch_to_string(stoichiometry):
    string = ""
    for idx, metal in enumerate(metals):
        string += f"{metal}$_{{{stoichiometry[idx]:.2f}}}$"
    return string

def stoch_to_string_exclusive(stoichiometry):
    string = ""
    for idx, metal in enumerate(metals):
        if stoichiometry[idx] > 0:
            string += f"{metal}$_{{{stoichiometry[idx]:.2f}}}$"
    return string

def activity_plot(activity_dict, filename):
    fig, ax = plt.subplots(figsize = (8, 5))
    ax.plot(activity_dict["potential_range"], activity_dict["j_avg_list"],         label = "Activity estimate based on all on-top sites")
    ax.plot(activity_dict["potential_range"], activity_dict["j_avg_special_list"], label = "Activity estimate based on select \nnon-CO-poisoned on-top sites")
    
    # Set axis labels
    ax.set_xlabel('Potential [V]')
    ax.set_ylabel('Estimated activity')

    # Set y-axis to a logarithmic scale
    #ax.set_yscale('log')
    max_j = activity_dict["special_max_j"]
    ax.text(x=0.1, y=activity_dict["special_max_j"], s=
            "Surface stoichiometry \n"+
            stoch_to_string(activity_dict["stoichiometry"]) + "\n" +
            f"Maximum activity: {max_j:.1e}")

    ax.legend(loc = "upper right")
    plt.savefig("../Activity_Estimation/"+filename, dpi = 400, bbox_inches = "tight")
    fig.show()
    return None

#### Loading and plotting the estimated activities ####

# Function to parse a string into a list of floats
def parse_molar_fraction(s): # Helps loading the .csv files! Made with ChatGPT assistance
    # Remove brackets and split by spaces
    values = s.strip('[]').split()

    # Convert each value to float
    return [float(value) for value in values]

def load_csv_activity_data(filename):
    # Load CSV file into a pandas DataFrame
    df = pd.read_csv(filename)

    # Extract columns and convert them to lists
    molar_fractions = df['Molar_Fraction'].tolist()
    estimated_activities = df['Estimated_Activities'].tolist()
    estimated_max_eUs = df['Estimated_Max_eUs'].tolist()

    molar_fractions = np.array([parse_molar_fraction(molar_frac) for molar_frac in molar_fractions])

    return molar_fractions, estimated_activities, estimated_max_eUs

def remove_columns(matrix, columns_to_keep):
    return [list(row[i] for i in columns_to_keep) for row in matrix]

def keep_columns(matrix, columns_to_keep):
    return [list(row[i] for i in columns_to_keep) for row in matrix]

def make_empty_plot():
    fig, ax = plt.subplots(figsize = (6, 6))

    # Remove ticks, axis labels, and everything
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    return fig, ax

## All this code was written by Jack . Ref: https://seafile.erda.dk/seafile/d/8586692f13/files/?p=%2Fscripts%2F__init__.py

def count_elements(elements, n_elems):
	count = np.zeros(n_elems, dtype=int)
	for elem in elements:
		count[elem] += 1
	return count

def get_molar_fractions(step_size, n_elems, total=1., return_number_of_molar_fractions=False):
	'Get all molar fractions with the given step size'
	
	interval = int(total/step_size)
	n_combs = scipy.special.comb(n_elems+interval-1, interval, exact=True)
	
	if return_number_of_molar_fractions:
		return n_combs
		
	counts = np.zeros((n_combs, n_elems), dtype=int)

	for i, comb in enumerate(it.combinations_with_replacement(range(n_elems), interval)):
		counts[i] = count_elements(comb, n_elems)

	return counts*step_size

def get_composition(f, metals, return_latex=False, saferound=True):
	
	# Make into numpy and convert to atomic percent
	f = np.asarray(f)*100
	
	if saferound:
		# Round while maintaining the sum, the iteround module may need
		# to be installed manually from pypi: "pip3 install iteround"
		import iteround
		f = iteround.saferound(f, 0)
	
	if return_latex:
		# Return string in latex format with numbers as subscripts
		return ''.join(['$\\rm {0}_{{{1}}}$'.format(m,f0) for m,f0 in\
			zip(metals, map('{:.0f}'.format, f)) if float(f0) > 0.])
	else:
		# Return composition as plain text
		return ''.join([''.join([m, f0]) for m,f0 in\
			zip(metals, map('{:.0f}'.format, f)) if float(f0) > 0.])

n_metals = 3
def get_simplex_vertices(n_elems=n_metals):

	# Initiate array of vertice coordinates
	vertices = np.zeros((n_elems, n_elems-1))
	
	for idx in range(1, n_elems):
		
		# Get coordinate of the existing dimensions as the 
		# mean of the existing vertices
		vertices[idx] = np.mean(vertices[:idx], axis=0)
		
		# Get the coordinate of the new dimension by ensuring it has a unit 
		# distance to the first vertex at the origin 
		vertices[idx][idx-1] = (1 - np.sum(vertices[idx][:-1]**2))**0.5
		
	return vertices

def molar_fractions_to_cartesians(fs):
	
	# Make into numpy
	fs = np.asarray(fs)

	if fs.ndim == 1:
		fs = np.reshape(fs, (1, -1))

	# Get vertices of the multidimensional simplex
	n_elems = fs.shape[1]
	vertices = get_simplex_vertices(n_elems)	
	vertices_matrix = vertices.T
	
	# Get cartisian coordinates corresponding to the molar fractions
	return np.dot(vertices_matrix, fs.T)

def make_triangle_ticks(ax, start, stop, tick, n, offset=(0., 0.),
						fontsize=18, ha='center', tick_labels=True):
	r = np.linspace(0, 1, n+1)
	x = start[0] * (1 - r) + stop[0] * r
	x = np.vstack((x, x + tick[0]))
	y = start[1] * (1 - r) + stop[1] * r
	y = np.vstack((y, y + tick[1]))
	ax.plot(x, y, 'k', lw=1., zorder=0)
	
	if tick_labels:
	
		# Add tick labels
		for xx, yy, rr in zip(x[0], y[0], r):
			ax.text(xx+offset[0], yy+offset[1], f'{rr*100.:.0f}',
					fontsize=fontsize, ha=ha)

def make_ternary_contour_plot(fs, zs, ax, elems, filename, cmap='viridis', levels=30,
							  color_norm=None, filled=True, axis_labels=True,
							  n_ticks=5, tick_labels=True, corner_labels=True):

	# Get cartesian coordinates corresponding to the molar fractions
	xs, ys = molar_fractions_to_cartesians(fs)
	
	# Make contour plot
	if filled:
		ax.tricontourf(xs, ys, zs, levels=levels, cmap=cmap, norm=color_norm, zorder=0)#, vmin = 0, vmax = 5*10**6)
	else:
		ax.tricontour(xs, ys, zs, levels=levels, cmap=cmap, norm=color_norm, zorder=0)#, vmin = 0, vmax = 5*10**6)
    
	# Specify vertices as molar fractions
	fs_vertices = [[1., 0., 0.],
				   [0., 1., 0.],
				   [0., 0., 1.]]
	
	# Get cartesian coordinates of vertices
	xs, ys = molar_fractions_to_cartesians(fs_vertices)
	
	# Make ticks and tick labels on the triangle axes
	left, right, top = np.concatenate((xs.reshape(-1,1), ys.reshape(-1,1)), axis=1)
	
	tick_size = 0.025
	bottom_ticks = 0.8264*tick_size * (right - top)
	right_ticks = 0.8264*tick_size * (top - left)
	left_ticks = 0.8264*tick_size * (left - right)
    
	make_triangle_ticks(ax, right, left, bottom_ticks, n_ticks, offset=(0.03, -0.08), ha='center', tick_labels=tick_labels)
	make_triangle_ticks(ax, left, top, left_ticks, n_ticks, offset=(-0.03, -0.015), ha='right', tick_labels=tick_labels)
	make_triangle_ticks(ax, top, right, right_ticks, n_ticks, offset=(0.015, 0.02), ha='left', tick_labels=tick_labels)

	if axis_labels:	
		# Show axis labels (i.e. atomic percentages)
		ax.text(0.5, -0.12, f'{elems[0]} content (%)', rotation=0., fontsize=20, ha='center', va='center')
		ax.text(0.88, 0.5, f'{elems[1]} content (%)', rotation=-60., fontsize=20, ha='center', va='center')
		ax.text(0.12, 0.5, f'{elems[2]} content (%)', rotation=60., fontsize=20, ha='center', va='center')

	if corner_labels:
		
		# Define padding to put the text neatly
		pad = [[-0.13, -0.09],
			   [ 0.07, -0.09],
			   [-0.04,  0.09]]
		
		# Show the chemical symbol as text at each vertex
		for idx, (x, y, (dx, dy)) in enumerate(zip(xs, ys, pad)):
			if len(elems[idx]) > 2:
				ax.text(x+dx-0.25, y+dy, s=elems[idx], fontsize=24)
			else:
				ax.text(x+dx, y+dy, s=elems[idx], fontsize=24)
    
	plt.savefig(filename, dpi = 400, bbox_inches = "tight")
## Above code was written by Jack

def find_max_activity(molar_fractions, estimated_activities, estimated_max_eUs): # Written with ChatGPT assistance
    # Find the index of the maximum estimated activity
    max_activity_index = estimated_activities.index(max(estimated_activities))

    # Get the corresponding molar fraction and estimated_max_eUs
    max_activity_molar_fraction = molar_fractions[max_activity_index]
    max_activity_estimated_max_eUs = estimated_max_eUs[max_activity_index]

    # Print the results
    max_activity = max(estimated_activities)
    max_molar_fraction = max_activity_molar_fraction
    max_eU = max_activity_estimated_max_eUs

    print(f"The highest activity is: {max_activity:.2e}")
    print(f"The composition is: {max_molar_fraction}")
    print(f"The highest activity happens at the eU: {max_eU:.2f}")

    # Return the result as a tuple
    return max_activity, max_molar_fraction, max_eU

# Assuming you have three lists: molar_fractions_020, estimated_activities, estimated_max_eUs
def save_activities_csv(filename, molar_fractions, estimated_activities, estimated_max_eUs):
    # Specify the file name
    csv_file_name = "../Activity_Estimation/" + filename
    #filename = "molar_fractions_PtAgAu_activity.csv"

    # Combine the lists into rows
    data = zip(molar_fractions, estimated_activities, estimated_max_eUs)

    # Write to CSV file
    with open(csv_file_name, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Write header if needed
        csv_writer.writerow(['Molar_Fraction', 'Estimated_Activities', 'Estimated_Max_eUs'])
        
        # Write data
        csv_writer.writerows(data)

    print(f'Data has been saved to {csv_file_name}')
    return None

##### Counting activity estimation #####

# In: Surface + potential
# Out: Number of active and number of inactive and number of blocked sites
def counting_activity(surface, potential):
    H_G_values    = surface["H_G"] + potential
    COOH_G_values = surface["COOH_G"] - potential

    active = 0
    inactive = sum(1 for value in COOH_G_values.flatten() if value > 0) #Counts number of binding COOH binding sites
    blocked = 0

    for idx_x in range(dim_x):
        for idx_y in range(dim_y):
            # Check if any of the neighbouring hollow sites could house an H
            security_clearence = True # This being true means NO dangerous H neighbours
            for x_diff, y_diff in [(0, 0), (-1, 0), (0, -1)]:
                neighbour_H_G = H_G_values[idx_x+x_diff, idx_y+y_diff]
                if neighbour_H_G < 0:
                    security_clearence = False # This being false means there is at least one dangerous neighbour
                    continue
            
            if COOH_G_values[idx_x][idx_y] > 0:
                COOH_binds = False
            if COOH_G_values[idx_x][idx_y] < 0:
                COOH_binds = True

            if security_clearence and COOH_binds:
                # There is a safe on-top site and COOH binds - SIUUUUU
                active += 1
            if not security_clearence and COOH_binds: #COOH would bind if it could but it can't because of block
                # The site is blocked
                blocked += 1

    return active, inactive, blocked

def counting_activity_scan(surface, Vmin, Vmax, points):
    """Takes a surface and a potential, and returns counts of:
    Active_list: Sites where COOH binds and no neighbouring H
    Inactive_list: Sites where COOH wouldn't bind no matter the blocking
    Blocked_list: Sites where COOH would bind but blocked by H disprop
    It doesn't count Inactive AND blocked. That group is in inactive"""

    active_list = []
    inactive_list = []
    blocked_list = []
    potential_range = np.linspace(Vmin, Vmax, points)
    for potential in potential_range:
        active, inactive, blocked = counting_activity(surface, potential)
        active_list.append(active)
        inactive_list.append(inactive)
        blocked_list.append(blocked)
    return np.array(potential_range), np.array(active_list), np.array(inactive_list), np.array(blocked_list)

def counting_activity_plot(potential_range, active_list, inactive_list, blocked_list, split, filename):
    fig, ax = plt.subplots(figsize = (8, 5))
    n_sites = dim_x*dim_y #Change to fit dims
    ax.plot(potential_range, active_list/n_sites,   c = "green", label = "Active on-top sites")
    ax.plot(potential_range, inactive_list/n_sites, c = "grey", label = "Inactive on-top sites")
    ax.plot(potential_range, blocked_list/n_sites,  c = "r", label = "Blocked on-top sites")

    # Set the major ticks and tick labels
    ax.set_xticks([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xticklabels([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # Put the stoichiometry in there
    ax.text(x=0.10, y=0.95, s = stoch_to_string(split))

    # Set axis labels
    ax.set_xlabel('Potential [V]')
    ax.set_ylabel('Number of sites as a fraction of all on-top sites')

    # Set the grid lines
    ax.grid(which='both', linestyle=':', linewidth=0.5, color='gray')

    ax.legend()

    plt.savefig(filename, dpi = 400, bbox_inches = "tight")
    plt.show()
    return None

def counting_activity_plot(potential_range, active_list, inactive_list, blocked_list, specific_potential, split, filename):
    fig, ax = plt.subplots(figsize = (8, 5))
    n_sites = dim_x*dim_y #Change to fit dims
    ax.plot(potential_range, active_list/n_sites,   c = "green", label = "Active on-top sites")
    ax.plot(potential_range, inactive_list/n_sites, c = "grey", label = "Inactive on-top sites")
    ax.plot(potential_range, blocked_list/n_sites,  c = "r", label = "Blocked on-top sites")

    # Set the major ticks and tick labels
    ax.set_xticks     ([0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
    ax.set_xticklabels([0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
    ax.set_yticks     ([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    ax.set_ylim(-0.05, 1.05)

    # Put the stoichiometry in there
    ax.text(x=0.175, y=0.93, s = stoch_to_string(split))

    # Set axis labels
    ax.set_xlabel('Potential [V]')
    ax.set_ylabel('Number of sites as a fraction of all on-top sites')

    # Set the grid lines
    ax.grid(which='both', linestyle=':', linewidth=0.5, color='gray')

    # Make a line showing the potential at which the composition is the best
    ax.vlines(x = specific_potential, ymin = -0.1, ymax = 1.1, color = 'black', linestyle='dotted')
    ax.text(x = specific_potential+0.003, y = 0.71, s = f"$eU = {specific_potential:.2f}\,eV$")

    ax.legend(loc = "center right")

    plt.savefig(filename, dpi = 400, bbox_inches = "tight")
    plt.show()
    return None

def write_to_csv(file_name, column_names, *data_lists):
    # Zip all the data lists together
    data = zip(*list(data_lists))

    # Write to CSV
    with open(file_name, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write header
        csv_writer.writerow(column_names)

        # Write data
        csv_writer.writerows(data)

    print(f'Data has been written to {file_name}')

def load_max_counting_activity(filename):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Access the specific column
    active_list = df["Active"]

    optimal_index = active_list.idxmax()
    optimal_active_sites = active_list[optimal_index]
    optimal_split = [df["Ag"][optimal_index], df["Au"][optimal_index], df["Cu"][optimal_index], df["Pd"][optimal_index], df["Pt"][optimal_index]]
    return optimal_active_sites, optimal_split

def load_all_counting_activity(filename):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Access the specific column
    active_list = df["Active"]
    splits = [df["Ag"], df["Au"], df["Cu"], df["Pd"], df["Pt"]]
    splits = np.array(splits).T

    return active_list, splits

def power_puff(potential, active_fraction):
    """Calculates the expected power from the potential and the fraction of active sites"""
    Cathode_potential = 0.9 #Jack said, that it's said to be at 0.9 V currently
    Open_circuit_current = Cathode_potential - potential # Volt or eV, if you think about an electron moving over that voltage
    power_per_site = Open_circuit_current * active_fraction # Volt times 
    return power_per_site, Open_circuit_current

# Create a mask based on the condition
def select_metals_summing_to_one(splits, active_list, select_metals):
    mask = np.sum(splits[:, select_metals], axis=1) == 1.0
    splits_PtAgAu = splits[mask]
    active_list_PtAgAu = active_list[mask]
    splits_PtAgAu = keep_columns(splits_PtAgAu, select_metals)
    return active_list_PtAgAu, splits_PtAgAu

def select_metals_summing_to_one_special_needs(splits, active_list, select_metals, equal_metals):
    mask_sum = np.sum(splits[:, select_metals], axis=1) == 1.0
    mask_equal = splits[:, equal_metals[0]] == splits[:, equal_metals[1]]
    super_mask = mask_sum & mask_equal
    splits_PtAgAu = splits[super_mask]
    active_list_PtAgAu = active_list[super_mask]
    splits_PtAgAu = keep_columns(splits_PtAgAu, select_metals)
    return active_list_PtAgAu, splits_PtAgAu

#####

