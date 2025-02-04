from flask import Blueprint, request, jsonify
import pandas as pd
import scipy.stats as stats
import numpy as np
from app.logger import logger

# Define Blueprint for Two-Sample t-Test API
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
    """
    Extracts the first two available keys dynamically.
    """
    keys = list(json_data.keys())

    if len(keys) < 2:
        raise ValueError("JSON input must contain at least two groups.")

    group1_key, group2_key = keys[:2]  # Automatically select the first two keys
    group1, group2 = json_data[group1_key], json_data[group2_key]

    if not isinstance(group1, list) or not isinstance(group2, list):
        raise ValueError("Both groups must be lists of numerical values.")

    return {
        "data": {
            "group1": group1,
            "group2": group2
        },
        "group1_name": group1_key,
        "group2_name": group2_key
    }

def _read_file_input(file):
    """
    Reads data from a CSV or Excel file and extracts two columns.
    """
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.filename.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")

    if len(df.columns) < 2:
        raise ValueError("CSV/Excel file must contain at least two columns.")

    column_names = df.columns[:2]  # Automatically select the first two columns
    return {
        "data": {
            "group1": df[column_names[0]].dropna().tolist(),
            "group2": df[column_names[1]].dropna().tolist()
        },
        "group1_name": column_names[0],
        "group2_name": column_names[1]
    }

def calculate_and_format_two_sample_t_test(group1, group2, group1_name, group2_name, alternative='two-sided', confidence=0.95):
    """
    Perform two-sample t-test, normality test, and equal variance test, and format the output.
    """
    try:
        # Normality Test
        shapiro_group1 = stats.shapiro(group1)
        shapiro_group2 = stats.shapiro(group2)

        # Equal Variance Test (Levene's Test)
        equal_var_test = stats.levene(group1, group2)

        # Sample Statistics
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        sem1, sem2 = std1 / np.sqrt(len(group1)), std2 / np.sqrt(len(group2))
        mean_difference = mean1 - mean2

        # Welch's t-test (Unequal Variance)
        t_stat_separate, p_value_separate = stats.ttest_ind(group1, group2, equal_var=False)
        df_separate = stats.ttest_ind(group1, group2, equal_var=False).df

        # Student's t-test (Equal Variance)
        t_stat_pooled, p_value_pooled = stats.ttest_ind(group1, group2, equal_var=True)
        df_pooled = len(group1) + len(group2) - 2

        # Confidence Interval Calculation
        t_critical = stats.t.ppf(1 - (1 - confidence) / 2, df_separate)
        margin_of_error = t_critical * np.sqrt((std1 ** 2 / len(group1)) + (std2 ** 2 / len(group2)))
        lower_bound = mean_difference - margin_of_error
        upper_bound = mean_difference + margin_of_error

        return {
            "Data Source": "Provided Data",
            "Normality Test (Shapiro-Wilk)": {
                group1_name: {"P-Value": round(shapiro_group1.pvalue, 3)},
                group2_name: {"P-Value": round(shapiro_group2.pvalue, 3)}
            },
            "Equal Variance Test (Levene's Test)": {
                "P-Value": round(equal_var_test.pvalue, 3)
            },
            "Sample Statistics": [
                {"Group": group1_name, "N": len(group1), "Mean": round(mean1, 3), "Std Dev": round(std1, 3), "SEM": round(sem1, 3)},
                {"Group": group2_name, "N": len(group2), "Mean": round(mean2, 3), "Std Dev": round(std2, 3), "SEM": round(sem2, 3)}
            ],
            "Difference of Means": round(mean_difference, 3),
            "Equal Variances Assumed (Student's t-test)": {
                "t": round(t_stat_pooled, 3),
                "df": df_pooled,
                "95% CI": [round(lower_bound, 3), round(upper_bound, 3)],
                "Two-tailed P-value": round(p_value_pooled, 5)
            },
            "Equal Variances Not Assumed (Welch's t-test)": {
                "t": round(t_stat_separate, 3),
                "df": round(df_separate, 3),
                "95% CI": [round(lower_bound, 3), round(upper_bound, 3)],
                "Two-tailed P-value": round(p_value_separate, 5)
            }
        }
    except Exception as e:
        raise ValueError(f"Error in two-sample t-test calculation: {str(e)}")

@two_sample_t_test_api.route('/two_sample_ttest', methods=['POST'])
def two_sample_ttest():
    """
    API endpoint for two-sample t-test.
    Accepts JSON or file input and returns statistical test results.
    """
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

        output_data = calculate_and_format_two_sample_t_test(
            groups["data"]["group1"],
            groups["data"]["group2"],
            groups["group1_name"],  # Corrected key reference
            groups["group2_name"],  # Corrected key reference
            alternative='two-sided',
            confidence=0.95
        )
        return jsonify(output_data), 200
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500
