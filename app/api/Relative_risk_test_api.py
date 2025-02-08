from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import logging
from scipy.stats import chi2_contingency
from datetime import datetime  # Import datetime module
from statsmodels.stats.power import NormalIndPower

# Initialize Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Flask Blueprint
relative_risk_api = Blueprint('relative_risk_api', __name__)

def read_json_input(json_data):
    """ Reads JSON input and converts it into a DataFrame. """
    try:
        df = pd.DataFrame(json_data['data'], columns=json_data['columns'], index=json_data['rows'])
        return df
    except Exception as e:
        raise ValueError(f"Error reading JSON input: {str(e)}")

def calculate_relative_risk(df, alpha=0.05):
    """ Calculates the Relative Risk, Chi-square, confidence intervals, and Power. """
    # Extract values from the contingency table
    a = df.iloc[0, 0]  # Treatment group with outcome
    b = df.iloc[0, 1]  # Treatment group without outcome
    c = df.iloc[1, 0]  # Control group with outcome
    d = df.iloc[1, 1]  # Control group without outcome

    # Calculate Proportions for cases (smokers) and controls (non-smokers)
    p_treatment = a / (a + b) if (a + b) != 0 else 0
    p_control = c / (c + d) if (c + d) != 0 else 0

    # Calculate Relative Risk
    relative_risk = p_treatment / p_control if p_control != 0 else float('inf')

    # Calculate Confidence Intervals for Relative Risk
    # Log-transformed CI for relative risk
    se_log_rr = np.sqrt(1/a + 1/b + 1/c + 1/d)
    ci_lower = np.exp(np.log(relative_risk) - 1.96 * se_log_rr)
    ci_upper = np.exp(np.log(relative_risk) + 1.96 * se_log_rr)

    # Perform Chi-square Test for independence
    chi2_stat, p_value, _, _ = chi2_contingency(df.values)
    
    # Degrees of Freedom for 2x2 table
    df_degrees_of_freedom = 1

    # Calculate Power of the Test
    # Effect size based on chi-square statistic and sample size
    effect_size = np.sqrt(chi2_stat / np.sum(df.values))  # Approximate effect size for power calculation
    power_analysis = NormalIndPower().power(effect_size=effect_size, nobs1=a + b, alpha=alpha)

    return relative_risk, ci_lower, ci_upper, p_treatment, p_control, chi2_stat, p_value, df_degrees_of_freedom, power_analysis

@relative_risk_api.route('/relative_risk', methods=['POST'])
def relative_risk():
    """
    API endpoint for Relative Risk Calculation.
    Accepts JSON input and returns Relative Risk and related metrics.
    """
    try:
        logger.info("Received request for Relative Risk calculation.")

        if not request.is_json:
            return jsonify({"error": "Invalid input format. Please provide JSON data."}), 400

        json_data = request.get_json()
        df = read_json_input(json_data)
        relative_risk, ci_lower, ci_upper, p_treatment, p_control, chi2_stat, p_value, df_degrees_of_freedom, power_analysis = calculate_relative_risk(df)

        # Get current date and time
        timestamp = datetime.now().strftime("%d %B %Y %H:%M:%S")

        # Conclusion based on p-value
        conclusion = "The likelihood of the outcome smoker is greater in the treatment group than in the control group." if p_value < 0.05 else "No significant difference observed."

        return jsonify({
            "Timestamp": timestamp,
            "Data Source": "Data 1 in Notebook1",
            "Contingency Table": df.round(3).to_dict(),
            "Relative Risk Value": round(relative_risk, 3),
            "Chi-square statistic": round(chi2_stat, 3),
            "Degrees of Freedom": df_degrees_of_freedom,
            "P-Value": round(p_value, 3),
            "95% Confidence Interval": [round(ci_lower, 3), round(ci_upper, 3)],
            "Power of Test": round(power_analysis, 3),
            "Conclusion": conclusion
        }), 200

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500
