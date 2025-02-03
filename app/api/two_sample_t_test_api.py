from flask import Blueprint, request, jsonify 
from app.utils import validate_two_sample_t_test_input
from app.logger import logger
import pandas as pd
import scipy.stats as stats
import numpy as np

# Define Blueprint for Two-Sample t-Test
two_sample_t_test_api = Blueprint('two_sample_t_test_api', __name__)

def read_input_data(json_data=None, file=None):
    """
    Read input data from JSON or file (CSV/Excel) and return as a dictionary.
    """
    try:
        if json_data:
            return _read_json_input(json_data)
        if file:
            return _read_file_input(file)
        raise ValueError("No valid input source provided.")
    except Exception as e:
        raise ValueError(f"Error processing input data: {str(e)}")

def _read_json_input(json_data):
    group1 = json_data.get("group1")
    group2 = json_data.get("group2")
    if not group1 or not group2:
        raise ValueError("JSON must contain 'group1' and 'group2' fields.")
    return {"group1": group1, "group2": group2}

def _read_file_input(file):
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.filename.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
    if 'group1' not in df.columns or 'group2' not in df.columns:
        raise ValueError("CSV/Excel file must contain 'group1' and 'group2' columns.")
    return {"group1": df['group1'].tolist(), "group2": df['group2'].tolist()}

def calculate_and_format_two_sample_t_test(group1, group2, alternative='two-sided', confidence=0.95):
    """
    Perform two-sample t-test and format the output in a structured JSON format.
    """
    try:
        # Separate Variance Calculation
        t_stat_separate, p_value_separate = stats.ttest_ind(group1, group2, equal_var=False)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        mean_difference = mean1 - mean2
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        std_err_separate = np.sqrt(var1 / len(group1) + var2 / len(group2))
        df_separate = len(group1) + len(group2) - 2

        # Pooled Variance Calculation
        pooled_var = ((len(group1) - 1) * var1 + (len(group2) - 1) * var2) / (len(group1) + len(group2) - 2)
        std_err_pooled = np.sqrt(pooled_var * (1 / len(group1) + 1 / len(group2)))
        t_stat_pooled = (mean1 - mean2) / std_err_pooled
        df_pooled = len(group1) + len(group2) - 2
        p_value_pooled = 2 * (1 - stats.t.cdf(np.abs(t_stat_pooled), df_pooled))

        # Confidence Interval Calculation
        t_critical = stats.t.ppf(1 - (1 - confidence) / 2, df_separate)
        margin_of_error_separate = t_critical * std_err_separate
        lower_bound_separate = mean_difference - margin_of_error_separate
        upper_bound_separate = mean_difference + margin_of_error_separate

        # Returning the formatted output in the required format
        return {
            "Ho": "Mean1 = Mean2",
            "H1": "Mean1 <> Mean2",
            "Grouping Variable": "LEADER$",
            "Variables": [
                {
                    "Variable": "MIL",
                    "LEADER$": "Islamic",
                    "N": len(group1),
                    "Mean": round(mean1, 3),
                    "Standard Deviation": round(np.std(group1, ddof=1), 3)
                },
                {
                    "Variable": "MIL",
                    "LEADER$": "Catholic",
                    "N": len(group2),
                    "Mean": round(mean2, 3),
                    "Standard Deviation": round(np.std(group2, ddof=1), 3)
                }
            ],
            "Separate Variance": [
                {
                    "Variable": "MIL",
                    "LEADER$": "Islamic",
                    "Mean Difference": round(mean_difference, 3),
                    "Lower Limit": round(lower_bound_separate, 3),
                    "Upper Limit": round(upper_bound_separate, 3),
                    "t": round(t_stat_separate, 3),
                    "df": round(df_separate, 3),
                    "p-value": round(p_value_separate, 3)
                }
            ],
            "Pooled Variance": [
                {
                    "Variable": "MIL",
                    "LEADER$": "Islamic",
                    "Mean Difference": round(mean_difference, 3),
                    "Lower Limit": round(lower_bound_separate, 3),
                    "Upper Limit": round(upper_bound_separate, 3),
                    "t": round(t_stat_pooled, 3),
                    "df": round(df_pooled, 3),
                    "p-value": round(p_value_pooled, 3)
                }
            ]
        }
    except Exception as e:
        raise ValueError(f"Error in two-sample t-test calculation: {str(e)}")

@two_sample_t_test_api.route('/two_sample_ttest', methods=['POST'])
def two_sample_ttest():
    try:
        logger.info("Received a request to perform Two-Sample t-test.")
        
        if request.is_json:
            data = request.get_json()
            groups = read_input_data(json_data=data)
        elif 'file' in request.files:
            file = request.files['file']
            groups = read_input_data(file=file)
        else:
            return jsonify({"error": "No data provided. Please provide either JSON or file input."}), 400
        
        validated_data = validate_two_sample_t_test_input(groups)
        output_data = calculate_and_format_two_sample_t_test(
            validated_data["data"]["group1"], validated_data["data"]["group2"],
            alternative=validated_data.get("alternative", "two-sided"),
            confidence=validated_data.get("confidence", 0.95)
        )
        return jsonify(output_data), 200
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500
