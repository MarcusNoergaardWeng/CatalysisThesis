import xgboost;
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition
import matplotlib.ticker as ticker
import time

#### FUNCTIONS FOR BAYESIAN OPTIMIZATION ####

def simulate_loss_right_corner(surface_stochiometry):
    ## surface_stochiometry is a 5-len list of probabilities to draw each metal
    surface_stochiometry = np.array(surface_stochiometry) / np.sum(surface_stochiometry)
    dim_x, dim_y, metals = 200, 200, ['Ag', 'Au', 'Cu', 'Pd', 'Pt']
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
    dim_x, dim_y, metals = 140, 140, ['Ag', 'Au', 'Cu', 'Pd', 'Pt']
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
    
    surf = {"atoms": surf_atoms,\
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