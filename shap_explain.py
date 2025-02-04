import shap
import numpy as np
import matplotlib.pyplot as plt

def explain_with_shap(model, data_sample):
    # Initialize SHAP explainer
    explainer = shap.Explainer(model, data_sample)

    # Get SHAP values for the sample
    shap_values = explainer(data_sample)

    # Plot SHAP values
    shap.summary_plot(shap_values, data_sample, show=False)
    plt.show()
