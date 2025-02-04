from flask import Blueprint, request, jsonify
from app.utils import validate_paired_t_test_input
from app.logger import logger
import pandas as pd
import scipy.stats as stats
import numpy as np

# Define Blueprint for Paired t-Test
paired_t_test_api = Blueprint('paired_t_test_api', __name__)

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
    except ValueError as ve:
        logger.error(f"ValueError: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise ValueError(f"Error processing input data: {str(e)}")

def _read_json_input(json_data):
    variables = list(json_data.keys())
    if len(variables) != 2:
        raise ValueError("JSON must contain exactly two variables.")
    
    before = json_data.get(variables[0])
    after = json_data.get(variables[1])
    
    if not isinstance(before, list) or not isinstance(after, list):
        raise ValueError("Both variables must be lists.")
    
    if not before or not after:
        raise ValueError("Both variables must contain data.")
    
    return {variables[0]: before, variables[1]: after}

def _read_file_input(file):
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.filename.endswith(('.xls', '.xlsx')): 
        df = pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
    
    variables = df.columns.tolist()
    if len(variables) != 2:
        raise ValueError("CSV/Excel file must contain exactly two columns.")
    
    df = df.dropna()
    before = df[variables[0]].tolist()
    after = df[variables[1]].tolist()
    
    return {variables[0]: before, variables[1]: after}

def calculate_paired_t_test(before, after):
    """
    Perform paired t-test and return a structured JSON response.
    """
    try:
        if len(before) != len(after):
            raise ValueError("Input lists must have the same length.")
        
        # Normality Test (Shapiro-Wilk)
        differences = np.array(after) - np.array(before)
        normality_p = stats.shapiro(differences)[1]
        normality_passed = normality_p > 0.05
        
        # Calculate statistics
        mean_before = np.mean(before)
        mean_after = np.mean(after)
        mean_difference = np.mean(differences)
        std_dev_before = np.std(before, ddof=1)
        std_dev_after = np.std(after, ddof=1)
        std_dev_diff = np.std(differences, ddof=1)
        sem_diff = std_dev_diff / np.sqrt(len(before))
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(before, after)
        df = len(before) - 1
        ci_low, ci_high = mean_difference - 1.96 * sem_diff, mean_difference + 1.96 * sem_diff
        one_tailed_p = p_value / 2

        return {
            "Test": "Paired t-test",
            "Normality Test (Shapiro-Wilk)": {
                "P-Value": round(normality_p, 3),
                "Passed": bool(normality_passed)  # Ensures JSON serializability
            },
            "Sample Statistics": {
                "Before Treatment": {
                    "N": len(before),
                    "Mean": round(mean_before, 3),
                    "Std Dev": round(std_dev_before, 3),
                },
                "After Treatment": {
                    "N": len(after),
                    "Mean": round(mean_after, 3),
                    "Std Dev": round(std_dev_after, 3),
                },
                "Difference": {
                    "Mean Difference": round(mean_difference, 3),
                    "Std Dev": round(std_dev_diff, 3),
                    "SEM": round(sem_diff, 3)
                }
            },
            "T-Test Results": {
                "t-Statistic": round(t_stat, 3),
                "Degrees of Freedom": df,
                "95% Confidence Interval": [round(ci_low, 3), round(ci_high, 3)],
                "Two-Tailed P-Value": round(p_value, 5),
                "One-Tailed P-Value": round(one_tailed_p, 5)
            }
        }
    except Exception as e:
        logger.error(f"Error in paired t-test calculation: {str(e)}")
        raise ValueError(f"Error in paired t-test calculation: {str(e)}")

@paired_t_test_api.route('/paired_ttest', methods=['POST'])
def paired_ttest():
    try:
        if request.is_json:
            data = request.get_json()
            groups = read_input_data(json_data=data)
        elif 'file' in request.files:
            file = request.files['file']
            groups = read_input_data(file=file)
        else:
            return jsonify({"error": "No data provided. Please provide either JSON or file input."}), 400
        
        before_label, after_label = list(groups.keys())
        output_data = calculate_paired_t_test(groups[before_label], groups[after_label])
        return jsonify(output_data), 200
    except ValueError as ve:
        logger.error(f"ValueError: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "An internal error occurred."}), 500
