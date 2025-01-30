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
    before = json_data.get("before")
    after = json_data.get("after")
    if not before or not after:
        raise ValueError("JSON must contain 'before' and 'after' fields.")
    return {"before": before, "after": after}

def _read_file_input(file):
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.filename.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
    if 'before' not in df.columns or 'after' not in df.columns:
        raise ValueError("CSV/Excel file must contain 'before' and 'after' columns.")
    return {"before": df['before'].tolist(), "after": df['after'].tolist()}

def calculate_and_format_paired_t_test(before, after):
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

        return {
            "Hypothesis Testing": "Paired t-test",
            "H0": "Mean Difference = 0",
            "H1": "Mean Difference > 0",
            "Variables": [
                {"Variable": "SYSBP_BEFORE", "N": len(before), "Mean": round(mean_before, 3)},
                {"Variable": "SYSBP_AFTER", "N": len(after), "Mean": round(mean_after, 3)}
            ],
            "Results": [
                {
                    "Variable": "SYSBP_BEFORE",
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
            groups = read_input_data(json_data=data)
        elif 'file' in request.files:
            file = request.files['file']
            groups = read_input_data(file=file)
        else:
            return jsonify({"error": "No data provided. Please provide either JSON or file input."}), 400
        
        
        validated_data = validate_paired_t_test_input(groups)
        output_data = calculate_and_format_paired_t_test(
            validated_data["data"]["before"], validated_data["data"]["after"]
        )
        return jsonify(output_data), 200
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500
