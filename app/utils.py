import pandas as pd
from scipy.stats import norm
import math

# ---------------------------------------------
# Input Data Handling
# ---------------------------------------------
def read_input_data(json_data=None, file=None):
    """
    Read input data from JSON or file (CSV/Excel) and return as a pandas DataFrame.
    :param json_data: JSON data from POST request (optional)
    :param file: file object (CSV/Excel) (optional)
    :return: DataFrame
    """
    try:
        if json_data:
            # Handle JSON input
            groups = json_data.get("groups")
            if not groups or len(groups) != 2:
                raise ValueError("Input JSON must contain exactly two groups with size and proportion.")
            
            return pd.DataFrame(groups, columns=["size", "proportion"])
        
        if file:
            # Handle file input
            if file.filename.endswith('.csv'):
                return pd.read_csv(file)
            elif file.filename.endswith(('.xls', '.xlsx')):
                return pd.read_excel(file)
            else:
                raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
        
        raise ValueError("No valid input source provided.")
    
    except Exception as e:
        raise ValueError(f"Error processing input data: {str(e)}")


def validate_two_sample_input(data):
    """
    Validate and process the data required for a Two-Sample Z-Test.
    :param data: pandas DataFrame with input data
    :return: dictionary with required fields
    """
    # Required columns for two-sample Z-test
    required_columns = ['dataset_id', 'mean1', 'mean2', 'std1', 'std2', 'n1', 'n2']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Input file must contain the following columns: {', '.join(required_columns)}")
    
    if len(data) < 1:
        raise ValueError("Input file must contain at least one row of data.")
    
    row = data.iloc[0]
    return {
        'dataset_id': row['dataset_id'],
        'mean1': float(row['mean1']),
        'mean2': float(row['mean2']),
        'std1': float(row['std1']),
        'std2': float(row['std2']),
        'n1': int(row['n1']),
        'n2': int(row['n2']),
    }

def validate_paired_t_test_input(groups):
    """
    Validate the input data for the paired t-test.
    Ensures the 'before' and 'after' data are present and have the same length.
    """
    before = groups.get('before')
    after = groups.get('after')
    
    # Check that 'before' and 'after' are lists and have the same length
    if not isinstance(before, list) or not isinstance(after, list):
        raise ValueError("Both 'before' and 'after' must be lists.")
    
    if len(before) != len(after):
        raise ValueError("'before' and 'after' lists must have the same length.")
    
    return {"data": {"before": before, "after": after}}

def validate_two_sample_t_test_input(data):
    """
    Validates input data for a two-sample t-test.
    """
    try:
        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary.")
        
        if "group1" not in data or "group2" not in data:
            raise ValueError("Missing required fields: 'group1' and 'group2'.")
        
        group1, group2 = data["group1"], data["group2"]
        
        if not isinstance(group1, list) or not isinstance(group2, list):
            raise ValueError("Both 'group1' and 'group2' must be lists of numbers.")
        
        if len(group1) < 2 or len(group2) < 2:
            raise ValueError("Each group must contain at least two values.")
        
        if not all(isinstance(x, (int, float)) for x in group1 + group2):
            raise ValueError("All values in 'group1' and 'group2' must be numeric.")
        
        alternative = data.get("alternative", "two-sided")
        if alternative not in ["two-sided", "greater", "less"]:
            raise ValueError("Invalid alternative hypothesis. Choose from 'two-sided', 'greater', or 'less'.")
        
        confidence = data.get("confidence", 0.95)
        if not (0 < confidence < 1):
            raise ValueError("Confidence level must be between 0 and 1.")
        
        return {"data": {"group1": group1, "group2": group2}, "alternative": alternative, "confidence": confidence}
    
    except Exception as e:
        raise ValueError(f"Validation error: {str(e)}")


# ---------------------------------------------
# Output Data Preparation
# ---------------------------------------------
def prepare_output_data(results, groups=None, confidence_level=None):
    """
    Convert test results into a JSON-friendly format for output.
    :param results: Test results dictionary
    :param groups: DataFrame with groups data
    :param confidence_level: Confidence interval as a string (optional)
    :return: JSON-friendly dictionary
    """
    output = {
        "results": results
    }

    if groups is not None:
        output["groups"] = [
            {
                "group": 1,
                "size": groups.iloc[0]["size"],
                "mean": groups.iloc[0].get("mean"),
                "std": groups.iloc[0].get("std"),
            },
            {
                "group": 2,
                "size": groups.iloc[1]["size"],
                "mean": groups.iloc[1].get("mean"),
                "std": groups.iloc[1].get("std"),
            }
        ]
    
    if confidence_level is not None:
        output["confidence_level"] = f"{confidence_level * 100}%"
    
    return output

# ---------------------------------------------
# Statistical Calculation Helpers
# ---------------------------------------------
def calculate_two_sample_z_test(mean1, mean2, std1, std2, n1, n2):
    """
    Perform a Two-Sample Z-Test.
    :param mean1: Mean of sample 1
    :param mean2: Mean of sample 2
    :param std1: Standard deviation of sample 1
    :param std2: Standard deviation of sample 2
    :param n1: Sample size of sample 1
    :param n2: Sample size of sample 2
    :return: Z-Score, P-Value, and Confidence Interval
    """
    # Calculate pooled standard error
    pooled_se = math.sqrt((std1 ** 2 / n1) + (std2 ** 2 / n2))
    z_score = (mean1 - mean2) / pooled_se

    # Two-tailed p-value
    p_value = 2 * (1 - norm.cdf(abs(z_score)))

    # Confidence interval
    confidence_interval = (
        (mean1 - mean2) - 1.96 * pooled_se,
        (mean1 - mean2) + 1.96 * pooled_se
    )

    return {
        "z_score": z_score,
        "p_value": p_value,
        "confidence_interval": confidence_interval
    }
