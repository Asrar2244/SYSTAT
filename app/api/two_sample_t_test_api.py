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
        t_stat, p_value = stats.ttest_ind(group1, group2, alternative=alternative, equal_var=False)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        mean_difference = mean1 - mean2
        std_err = np.sqrt(np.var(group1, ddof=1) / len(group1) + np.var(group2, ddof=1) / len(group2))
        df = len(group1) + len(group2) - 2
        
        # Confidence Interval Calculation
        t_critical = stats.t.ppf(1 - (1 - confidence) / 2, df)
        margin_of_error = t_critical * std_err
        lower_bound = mean_difference - margin_of_error
        upper_bound = mean_difference + margin_of_error
        
        return {
            "Hypothesis Testing": "Two-Sample t-test",
            "H0": "Mean Difference = 0",
            "H1": f"Mean Difference {alternative.replace('-', ' ')} 0",
            "Variables": [
                {"Variable": "Group 1", "N": len(group1), "Mean": round(mean1, 3)},
                {"Variable": "Group 2", "N": len(group2), "Mean": round(mean2, 3)}
            ],
            "Results": [
                {
                    "Mean Difference": round(mean_difference, 3),
                    "Standard Error": round(std_err, 3),
                    "t": round(t_stat, 3),
                    "df": df,
                    "p-Value": round(p_value, 3),
                    "Confidence Interval": {
                        "Lower Bound": round(lower_bound, 3),
                        "Upper Bound": round(upper_bound, 3)
                    }
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
