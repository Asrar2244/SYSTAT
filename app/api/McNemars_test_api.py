from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import logging
from statsmodels.stats.contingency_tables import mcnemar
import scipy.stats as stats
from datetime import datetime  # Import datetime module

# Initialize Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Flask Blueprint
mcnemar_api = Blueprint('mcnemar_api', __name__)

def read_json_input(json_data):
    """ Reads JSON input and converts it into a DataFrame. """
    try:
        df = pd.DataFrame(json_data['data'], columns=json_data['columns'], index=json_data['rows'])
        return df
    except Exception as e:
        raise ValueError(f"Error reading JSON input: {str(e)}")

def calculate_odds_ratio_and_diff(df):
    """ Calculates the odds ratio, relative difference, and confidence intervals. """
    b, c = df.iloc[0, 1], df.iloc[1, 0]
    a, d = df.iloc[0, 0], df.iloc[1, 1]
    
    # Calculate Proportions for cases and controls
    p_case = a / (a + b) if (a + b) != 0 else 0
    p_control = d / (c + d) if (c + d) != 0 else 0
    
    # Calculate odds ratio and confidence intervals
    if b == 0 or c == 0:
        odds_ratio = float('inf')  # If any value is 0, odds ratio is undefined
        ci_lower, ci_upper = None, None
    else:
        odds_ratio = b / c
        se = np.sqrt(1/b + 1/c)  # Standard error
        ci_lower = np.exp(np.log(odds_ratio) - 1.96 * se)
        ci_upper = np.exp(np.log(odds_ratio) + 1.96 * se)
    
    # Calculate Relative Difference
    relative_diff = (p_case - p_control) / p_control * 100 if p_control != 0 else float('inf')
    
    return round(odds_ratio, 6), round(ci_lower, 6), round(ci_upper, 6), round(p_case, 6), round(p_control, 6), round(relative_diff, 6)

def perform_mcnemar_test(df):
    """ Performs McNemar's Test using statsmodels library. """
    try:
        if df.shape != (2, 2):
            raise ValueError("McNemar's test requires a 2x2 contingency table.")

        # McNemar's chi-square test (without Yates' correction)
        chi2_stat, p_value = mcnemar(df, exact=False, correction=False), None
        p_value = 1 - stats.chi2.cdf(chi2_stat.statistic, df=1)

        # Exact McNemar test (binomial test)
        exact_p_value = mcnemar(df, exact=True).pvalue

        # Get current date and time
        timestamp = datetime.now().strftime("%d %B %Y %H:%M:%S")

        # Compute odds ratio, confidence intervals, proportions, and relative difference
        odds_ratio, ci_lower, ci_upper, p_case, p_control, relative_diff = calculate_odds_ratio_and_diff(df)

        # Conclusion
        conclusion = "Significant difference between categories (P < 0.05)." if p_value < 0.05 else "No significant difference observed."

        return {
            "McNemar's Test": timestamp,  # Dynamic Timestamp
            "Data Source": "Data 1 in Notebook1",
            "Yates continuity correction": "Not applied to calculations.",
            "Chi-square": round(chi2_stat.statistic, 3),
            "Degrees of Freedom": 1,
            "P-Value": round(p_value, 5),
            "Exact McNemar Significance Probability": round(exact_p_value, 5),
            "Odds Ratio": {
                "Value": odds_ratio,
                "95% Confidence Interval": [ci_lower, ci_upper]
            },
            "Proportions": {
                "Cases Proportion": p_case,
                "Controls Proportion": p_control
            },
            "Relative Difference (%)": relative_diff,
            "Observed Counts": df.round(3).to_dict(),
            "Conclusion": conclusion
        }
    except Exception as e:
        raise ValueError(f"Error in McNemar's test calculation: {str(e)}")

@mcnemar_api.route('/mcnemar_test', methods=['POST'])
def mcnemar_test():
    """
    API endpoint for McNemar's Test.
    Accepts JSON input and returns test results.
    """
    try:
        logger.info("Received request for McNemar's test.")

        if not request.is_json:
            return jsonify({"error": "Invalid input format. Please provide JSON data."}), 400

        json_data = request.get_json()
        df = read_json_input(json_data)
        result = perform_mcnemar_test(df)

        return jsonify(result), 200
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500
