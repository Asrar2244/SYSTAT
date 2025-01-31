from flask import Blueprint, request, jsonify
from app.utils import validate_paired_t_test_input
from app.logger import logger
import pandas as pd
import scipy.stats as stats

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
    except Exception as e:
        raise ValueError(f"Error processing input data: {str(e)}")

def _read_json_input(json_data):
    variables = list(json_data.keys())
    if len(variables) != 2:
        raise ValueError("JSON must contain exactly two variables.")
    
    before = json_data.get(variables[0])
    after = json_data.get(variables[1])

    # Debug log to check input data
    logger.info(f"Received JSON data: {json_data}")
    
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
    
    before = df[variables[0]].tolist()
    after = df[variables[1]].tolist()

    # Debug log to check input data
    logger.info(f"Received file data: {df.head()}")
    
    if not isinstance(before, list) or not isinstance(after, list):
        raise ValueError("Both variables must be lists.")
    
    return {variables[0]: before, variables[1]: after}

def calculate_and_format_paired_t_test(before, after, before_label, after_label):
    """
    Perform paired t-test and format the output in a structured JSON format.
    """
    try:
        t_stat, p_value = stats.ttest_rel(before, after)
        mean_before, mean_after = sum(before) / len(before), sum(after) / len(after)
        mean_difference = mean_before - mean_after
        standard_deviation = stats.tstd([b - a for b, a in zip(before, after)])
        confidence_bound = mean_difference - (1.96 * (standard_deviation / (len(before) ** 0.5)))
        df = len(before) - 1

       # Structuring the result output dynamically
        return {
            "Hypothesis Testing": "Paired t-test",
            "H0": "Mean Difference = 0",
            "H1": "Mean Difference > 0",
            "Variables": [
                {"Variable": before_label, "N": len(before), "Mean": round(sum(before) / len(before), 3)},
                {"Variable": after_label, "N": len(after), "Mean": round(sum(after) / len(after), 3)}
            ],
            "Results": [
                {
                      "Variable": f"{before_label}  {after_label}", 
                    "Mean Difference": round(mean_difference, 3),
                    "95% Confidence Bound": round(confidence_bound, 3),
                    "Standard Deviation of Difference": round(standard_deviation, 3),
                    "t": round(t_stat, 3),
                    "df": df,
                    "p-Value": round(p_value, 3)
                }
            ]
        }
    except Exception as e:
        raise ValueError(f"Error in paired t-test calculation: {str(e)}")

@paired_t_test_api.route('/paired_ttest', methods=['POST'])
def paired_ttest():
    try:
        logger.info("Received a request to perform Paired t-test.")
        
        if request.is_json:
            data = request.get_json()
            logger.info(f"Received JSON input: {data}")  # Log the received JSON
            groups = read_input_data(json_data=data)
        
        elif 'file' in request.files:
            file = request.files['file']
            logger.info(f"Received file input: {file.filename}")  # Log the received file
            groups = read_input_data(file=file)
        
        else:
            return jsonify({"error": "No data provided. Please provide either JSON or file input."}), 400
        
        # Extract the labels of the two variables dynamically
        before_label, after_label = list(groups.keys())
        logger.info(f"before_label: {before_label}, after_label: {after_label}")  # Log the variable labels
        
        # Log the structure of the data
        logger.info(f"Groups data: {groups}")
        
        # Bypass validation temporarily
        # Check if the data is in list form (which is expected)
        if not isinstance(groups[before_label], list) or not isinstance(groups[after_label], list):
            return jsonify({"error": f"Both '{before_label}' and '{after_label}' must be lists."}), 400
        
        # Perform paired t-test directly without extra validation step
        output_data = calculate_and_format_paired_t_test(
            groups[before_label], groups[after_label], before_label, after_label
        )
        
        return jsonify(output_data), 200
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500 