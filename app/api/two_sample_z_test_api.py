import math
from scipy.stats import norm
from flask import Blueprint, request, jsonify
from app.utils import read_input_data, prepare_output_data
from app.logger import logger

# Initialize the Blueprint
two_sample_z_test = Blueprint('two_sample_z_test', __name__)

@two_sample_z_test.route('/', methods=['POST'])
def two_sample_z_test_func():
    try:

         # Log the request
        logger.info("Received a request to perform two sample Z-test.")

        # Check if JSON or file input is provided
        if request.is_json:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No input data provided."}), 400
            dataset_id = data['dataset_id']
            mean1 = float(data['mean1'])
            mean2 = float(data['mean2'])
            std1 = float(data['std1'])
            std2 = float(data['std2'])
            n1 = int(data['n1'])
            n2 = int(data['n2'])
        elif 'file' in request.files:
            file = request.files['file']
            data = read_input_data(file)  # Assuming this function processes the uploaded file
            dataset_id = data['dataset_id']
            mean1 = float(data['mean1'])
            mean2 = float(data['mean2'])
            std1 = float(data['std1'])
            std2 = float(data['std2'])
            n1 = int(data['n1'])
            n2 = int(data['n2'])
        else:
            return jsonify({"error": "No data provided. Please provide either JSON or file input."}), 400

        # Validate required parameters
        if n1 <= 0 or n2 <= 0 or std1 <= 0 or std2 <= 0:
            return jsonify({"error": "Sample sizes and standard deviations must be greater than zero."}), 400

        # Perform Two-Sample Z-Test
        pooled_std = math.sqrt((std1 ** 2 / n1) + (std2 ** 2 / n2))
        z_score = (mean1 - mean2) / pooled_std
        p_value = 2 * (1 - norm.cdf(abs(z_score)))

        # Confidence interval
        mean_diff = mean1 - mean2
        ci_low = mean_diff - 1.96 * pooled_std
        ci_high = mean_diff + 1.96 * pooled_std

        # Prepare the results
        results = {
            "dataset_id": dataset_id,
            "z_score": z_score,
            "p_value": p_value,
            "mean_difference": mean_diff,
            "confidence_interval": {
                "lower_bound": ci_low,
                "upper_bound": ci_high
            }
        }

        # Process output data
        output_data = prepare_output_data(results)  # Assuming this function processes output data
        return jsonify(output_data), 200

    except Exception as e:
        logger.error(f"Error during two-sample Z-test: {str(e)}")
        return jsonify({"error": str(e)}), 500
