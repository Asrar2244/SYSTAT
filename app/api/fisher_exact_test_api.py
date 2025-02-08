from flask import Blueprint, request, jsonify
import pandas as pd
import scipy.stats as stats
import numpy as np
import logging

# Initialize Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Flask Blueprint for Fisher Exact Test
fisher_exact_api = Blueprint('fisher_exact_api', __name__)

def read_json_input(json_data):
    """
    Reads input JSON and converts it to a Pandas DataFrame.
    """
    try:
        df = pd.DataFrame(json_data['data'], columns=json_data['columns'], index=json_data['rows'])
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
    Generates a comment based on the p-value of the Fisher Exact test.
    """
    if p_value < 0.05:
        return "The proportion of observations in the different categories is significantly different from what is expected under random occurrence."
    else:
        return "There is no significant difference in the proportion of observations across categories."

def perform_fisher_exact_test(df):
    """
    Performs the Fisher Exact test on the given DataFrame.
    """
    try:
        observed = df.to_numpy()  # Convert DataFrame to NumPy array
        oddsratio, p_value = stats.fisher_exact(observed)  # Perform Fisher's Exact Test

        # Calculate expected counts (approximate using row and column totals)
        row_totals = observed.sum(axis=1)
        col_totals = observed.sum(axis=0)
        grand_total = observed.sum()
        expected = np.outer(row_totals, col_totals) / grand_total
        expected_df = pd.DataFrame(expected, index=df.index, columns=df.columns)

        # Calculate percentages
        row_percentages, col_percentages, total_percentages = calculate_percentages(df)

        # Generate comment based on p-value
        comment = generate_comment(p_value)

        return {
            "Data Source": "Data 1 in Notebook1",
            "Counts and Percentages": {
                "Observed Counts": df.to_dict(),
                "Expected Counts": expected_df.round(3).to_dict(),
                "% Row Total": row_percentages.to_dict(),
                "% Column Total": col_percentages.to_dict(),
                "% Total": total_percentages.to_dict()
            },
            "Fisher Exact Test Results": {
                "Odds Ratio": round(oddsratio, 3),
                "P-Value": round(p_value, 5),
                "Conclusion": comment
            }
        }
    except Exception as e:
        raise ValueError(f"Error in Fisher Exact test calculation: {str(e)}")

@fisher_exact_api.route('/fisher_exact_test', methods=['POST'])
def fisher_exact_test():
    """
    API endpoint for Fisher Exact Test.
    Accepts JSON input and returns test results.
    """
    try:
        logger.info("Received request for Fisher Exact test.")

        if not request.is_json:
            return jsonify({"error": "Invalid input format. Please provide JSON data."}), 400

        json_data = request.get_json()
        df = read_json_input(json_data)
        result = perform_fisher_exact_test(df)

        return jsonify(result), 200
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500
