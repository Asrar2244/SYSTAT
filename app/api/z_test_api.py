from flask import Blueprint, request, jsonify
from scipy.stats import norm
import math
from app.utils import read_input_data, prepare_output_data
from app.logger import logger

z_test_api = Blueprint('z_test_api', __name__)

@z_test_api.route('/', methods=['POST'])
def z_test():
    try:
        # Log the request
        logger.info("Received a request to perform Z-test.")

        # Check if JSON or file is provided
        if request.is_json:
            data = request.get_json()
            groups = read_input_data(json_data=data)
        elif 'file' in request.files:
            file = request.files['file']
            groups = read_input_data(file=file)
        else:
            return jsonify({"error": "No data provided. Please provide either JSON or file input."}), 400

        # Extract necessary values
        group_1 = groups.iloc[0]
        group_2 = groups.iloc[1]
        n1, p1 = group_1["size"], group_1["proportion"]
        n2, p2 = group_2["size"], group_2["proportion"]

        # Additional parameters
        alpha_value = float(request.form.get("alpha_value", 0.05))
        yates_correction = int(request.form.get("yates_correction", 0))
        confidence_interval = int(request.form.get("confidence_interval", 95))

        # Calculate Z-test metrics
        pooled_p = (n1 * p1 + n2 * p2) / (n1 + n2)
        standard_error = math.sqrt(pooled_p * (1 - pooled_p) * ((1 / n1) + (1 / n2)))
        z_score = (p1 - p2) / standard_error
        p_value = 2 * (1 - norm.cdf(abs(z_score)))

        # Confidence interval
        z_critical = norm.ppf(1 - (1 - confidence_interval / 100) / 2)
        margin_of_error = z_critical * standard_error
        confidence_interval_lower = (p1 - p2) - margin_of_error
        confidence_interval_upper = (p1 - p2) + margin_of_error

        # Power of the test (using alpha value)
        power = 1 - norm.cdf(z_critical - abs(z_score))

        # Conclusion
        conclusion = "There is a significant difference in the proportions." if p_value < alpha_value else "No significant difference in the proportions."

        # Prepare results for output
        results = {
            "yates_correction": yates_correction,
            "results": {
                "The difference_of_sample_proportions": p1 - p2,
                "The pooled_estimate for p": pooled_p,
                "standard_error of difference of sample proportions": standard_error,
                "z_score": z_score,
                "p_value": p_value,
                "confidence_interval": {
                    "lower_bound": confidence_interval_lower,
                    "upper_bound": confidence_interval_upper
                },
                "power_of_test": power,
                "conclusion": conclusion
            }
        }

        # Return JSON response
        output_data = prepare_output_data(results, groups, confidence_interval)
        return jsonify(output_data), 200

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500
