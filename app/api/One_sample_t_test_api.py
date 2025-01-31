from flask import Blueprint, request, jsonify
import pandas as pd
import scipy.stats as stats
import numpy as np
from app.logger import logger

# Define Blueprint
t_test_api = Blueprint("t_test_api", __name__)

# Helper function to read input data (from JSON or file)
def read_input_data(json_data=None, file=None, target_column=None):
    """
    Reads input data from JSON or a file (CSV/Excel) and extracts the target column.
    """
    try:
        if json_data:
            return _read_json_input(json_data)
        if file:
            return _read_file_input(file, target_column)
        raise ValueError("No valid input source provided.")
    except Exception as e:
        raise ValueError(f"Error processing input data: {str(e)}")


# Helper function to process JSON input
def _read_json_input(json_data):
    sample_data = json_data.get("sample")
    if not sample_data:
        raise ValueError("JSON must contain a 'sample' field with data.")
    return sample_data


# Helper function to process file input (CSV/Excel)
def _read_file_input(file, target_column):
    if not target_column:
        raise ValueError("Target column must be specified when using file input.")
    
    # Read the file based on its extension
    if file.filename.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.filename.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
    
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in dataset.")
    
    return df[target_column].dropna().tolist()


# Function to calculate One-Sample t-test
def calculate_one_sample_t_test(sample, population_mean, alternative="two-sided", confidence=0.95):
    """
    Perform a one-sample t-test and return the results in a structured JSON format.
    """
    try:
        sample_size = len(sample)
        if sample_size < 2:
            raise ValueError("Sample size must be greater than one.")
        
        # Calculate sample statistics
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)
        df = sample_size - 1
        t_stat, p_value = stats.ttest_1samp(sample, population_mean)

        # Adjust p-value for one-tailed tests
        if alternative == "greater":
            p_value = 1 - stats.t.cdf(t_stat, df)
        elif alternative == "less":
            p_value = stats.t.cdf(t_stat, df)
        else:
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))  # Two-tailed test

        # Confidence Interval Calculation
        if alternative in ["greater", "less"]:
            t_critical = stats.t.ppf(1 - confidence, df)
        else:
            t_critical = stats.t.ppf(1 - (1 - confidence) / 2, df)
        
        margin_of_error = t_critical * (sample_std / np.sqrt(sample_size))
        lower_bound = sample_mean - margin_of_error
        upper_bound = sample_mean + margin_of_error
        
        # Return results as structured JSON
        return {
            "Hypothesis Testing": "One-Sample t-test",
            "H0": f"Mean = {population_mean}",
            "H1": f"Mean {alternative.replace('-', ' ')} {population_mean}",
            "Sample Statistics": {
                "Sample Size": sample_size,
                "Sample Mean": round(sample_mean, 3),
                "Sample Std Dev": round(sample_std, 3),
                "Degrees of Freedom": df,
                "t-Statistic": round(t_stat, 3),
                "p-Value": round(p_value, 3),
                "Confidence Interval": {
                    "Lower Bound": round(lower_bound, 3),
                    "Upper Bound": round(upper_bound, 3),
                    "Confidence Level": confidence * 100  # Convert to percentage for clarity
                }
            }
        }
    except Exception as e:
        raise ValueError(f"Error in one-sample t-test calculation: {str(e)}")


# Route to handle One-Sample t-test requests
@t_test_api.route("/one_sample_ttest", methods=["POST"])
def one_sample_ttest():
    try:
        logger.info("Received a request to perform One-Sample t-test.")
        
        # Check if the request is JSON
        if request.is_json:
            data = request.get_json()
            logger.debug(f"Received JSON data: {data}")
            sample = read_input_data(json_data=data)
            population_mean = float(data.get("population_mean", 0))
            alternative = data.get("alternative", "two-sided")
            
            # Extract confidence level from JSON (defaults to 0.95 if not provided)
            confidence = data.get("confidence_level", 0.95)
            logger.debug(f"Extracted confidence value: {confidence}")
        
        # Check if the request contains a file
        elif "file" in request.files:
            file = request.files["file"]
            population_mean = float(request.form.get("population_mean", 0))
            alternative = request.form.get("alternative", "two-sided")
            
            # Extract confidence level from form data
            confidence = request.form.get("confidence_level", 0.95)
            logger.debug(f"Extracted confidence value from form: {confidence}")
            
            target_column = request.form.get("target_column")
            sample = read_input_data(file=file, target_column=target_column)
        
        else:
            return jsonify({"error": "No data provided. Please provide either JSON or file input."}), 400
        
        # Log the confidence level being used
        logger.debug(f"Using Confidence Level: {confidence}")
        
        # Perform the t-test with the extracted data and confidence level
        output_data = calculate_one_sample_t_test(sample, population_mean, alternative, float(confidence))
        return jsonify(output_data), 200

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500
