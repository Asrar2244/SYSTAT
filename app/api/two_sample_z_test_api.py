import math
import pandas as pd
from scipy.stats import norm
from flask import Blueprint, request, jsonify
from app.utils import read_input_data, prepare_output_data
from app.logger import logger

# Initialize the Blueprint
two_sample_z_test = Blueprint('two_sample_z_test', __name__)

@two_sample_z_test.route('/', methods=['POST'])
def two_sample_z_test_func():
    try:
        logger.info("Received a request to perform two-sample Z-test.")

        # Read input data
        if request.is_json:
            data = request.get_json()
        elif 'file' in request.files:
            file = request.files['file']
            data = read_input_data(file)
        else:
            return jsonify({"error": "No data provided. Please provide either JSON or file input."}), 400

        # Extract parameters
        column = data['column']  # Column name for life expectancy
        group_col = data['group_column']  # Column indicating 'Developed' or 'Emerging'
        std1 = float(data['std1'])  # User-specified standard deviation for group 1
        std2 = float(data['std2'])  # User-specified standard deviation for group 2
        confidence = float(data.get('confidence', 0.95))  # Default 95%
        alternative = data.get('alternative', 'NE').upper()  # NE, LT, GT

        # Validate standard deviations
        if std1 <= 0 or std2 <= 0:
            return jsonify({"error": "Standard deviations must be greater than zero."}), 400

        # Convert data to DataFrame
        df = pd.DataFrame(data['data'])

        # Compute means and counts for the two groups
        grouped = df.groupby(group_col)[column].agg(['mean', 'count', 'std'])
        if len(grouped) != 2:
            return jsonify({"error": "Invalid grouping variable. Ensure exactly two groups."}), 400

        (mean1, n1), (mean2, n2) = grouped[['mean', 'count']].values

        # Compute Z-score
        pooled_std = math.sqrt((std1 ** 2 / n1) + (std2 ** 2 / n2))
        z_score = (mean1 - mean2) / pooled_std

        # Compute p-value based on alternative hypothesis
        if alternative == 'NE':  # Two-tailed test
            p_value = 2 * (1 - norm.cdf(abs(z_score)))
        elif alternative == 'LT':  # One-tailed test (mean1 < mean2)
            p_value = norm.cdf(z_score)
        else:  # One-tailed test (mean1 > mean2)
            p_value = 1 - norm.cdf(z_score)

        # Compute confidence interval
        z_critical = norm.ppf(1 - (1 - confidence) / 2)
        mean_diff = mean1 - mean2
        ci_low = mean_diff - z_critical * pooled_std
        ci_high = mean_diff + z_critical * pooled_std

        # Prepare results
        results = {
            "hypothesis": "Ho: Mean1 = Mean2 vs H1: Mean1 {} Mean2".format(
                "<>" if alternative == "NE" else ("<" if alternative == "LT" else ">")
            ),
            "grouping_variable": group_col,
            "sd1": std1,
            "sd2": std2,
            "summary": {
                "Group1": {"N": n1, "Mean": mean1},
                "Group2": {"N": n2, "Mean": mean2}
            },
            "confidence_interval": {
                "confidence_level": confidence,
                "mean_difference": mean_diff,
                "lower_bound": ci_low,
                "upper_bound": ci_high
            },
            "z_score": z_score,
            "p_value": p_value
        }

        return jsonify(results), 200

    except Exception as e:
        logger.error(f"Error during two-sample Z-test: {str(e)}")
        return jsonify({"error": str(e)}), 500
