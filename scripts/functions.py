import xgboost;
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition
import matplotlib.ticker as ticker
import time

figure_folder = "../figures/DeltaG_models"

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
