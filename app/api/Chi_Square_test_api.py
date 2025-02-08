from flask import Blueprint, request, jsonify
import pandas as pd
import scipy.stats as stats
import numpy as np
import logging

# Initialize Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Flask Blueprint for Chi-Square Test
chi_square_api = Blueprint('chi_square_api', __name__)

def read_json_input(json_data):
    """
    Reads input JSON and converts it to a Pandas DataFrame.
    """
    try:
        df = pd.DataFrame(json_data)
        return df
    except Exception as e:
        raise ValueError(f"Error reading JSON input: {str(e)}")

def calculate_percentages(observed):
    """
    Calculates row-wise, column-wise, and total-wise percentages.
    """
    row_totals = observed.sum(axis=1)  # Sum of each row
    col_totals = observed.sum(axis=0)  # Sum of each column
    grand_total = observed.sum().sum()  # Overall total count

    # Calculate percentages
    row_percentages = (observed.div(row_totals, axis=0) * 100).round(3)
    col_percentages = (observed.div(col_totals, axis=1) * 100).round(3)
    total_percentages = (observed / grand_total * 100).round(3)

    return row_percentages, col_percentages, total_percentages

def generate_comment(p_value):
    """
    Generates a comment based on the p-value of the Chi-Square test.
    """
    if p_value < 0.05:
        return "The two characteristics are significantly related, meaning there is an association between the categories."
    else:
        return "There is no significant relationship between the two characteristics, meaning they are independent."

def perform_chi_square_test(df):
    """
    Performs the Chi-Square test for independence on the given DataFrame.
    """
    try:
        observed = df.to_numpy()  # Convert DataFrame to NumPy array
        chi2, p, dof, expected = stats.chi2_contingency(observed)  # Perform Chi-Square test

        # Compute test power (approximated)
        alpha = 0.05
        power = 1 - stats.chi2.cdf(chi2, dof, loc=0, scale=1)

        # Convert expected counts back to DataFrame
        expected_df = pd.DataFrame(expected, index=df.index, columns=df.columns)

        # Calculate percentages
        row_percentages, col_percentages, total_percentages = calculate_percentages(df)

        # Generate comment based on p-value
        comment = generate_comment(p)

        return {
            "Data Source": "Data 1 in Notebook1",
            "Counts and Percentages": {
                "Observed Counts": df.to_dict(),
                "Expected Counts": expected_df.round(3).to_dict(),
                "% Row Total": row_percentages.to_dict(),
                "% Column Total": col_percentages.to_dict(),
                "% Total": total_percentages.to_dict()
            },
            "Chi-Squared Tests for Independence": {
                "Pearson Chi-Square": round(chi2, 3),
                "Degrees of Freedom": dof,
                "P-Value": round(p, 5),
                "Test Power (alpha = 0.05)": round(power, 3),
                "Conclusion": comment
            }
        }
    except Exception as e:
        raise ValueError(f"Error in Chi-Square test calculation: {str(e)}")

@chi_square_api.route('/chi_square_test', methods=['POST'])
def chi_square_test():
    """
    API endpoint for Chi-Square Test.
    Accepts JSON input and returns test results.
    """
    try:
        logger.info("Received request for Chi-Square test.")

        if not request.is_json:
            return jsonify({"error": "Invalid input format. Please provide JSON data."}), 400

        json_data = request.get_json()
        df = read_json_input(json_data)
        result = perform_chi_square_test(df)

        return jsonify(result), 200
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500
