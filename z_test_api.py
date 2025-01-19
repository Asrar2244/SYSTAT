from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.stats import norm
import math

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

@app.route('/ztest', methods=['POST'])
def one_sample_z_test():
    try:
        # Extracting data from the request
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided."}), 400

        # Retrieve parameters from the input
        sample_mean = data.get("sample_mean")
        population_mean = data.get("population_mean")
        standard_deviation = data.get("standard_deviation")
        sample_size = data.get("sample_size")

        # Validate inputs
        if None in [sample_mean, population_mean, standard_deviation, sample_size]:
            return jsonify({"error": "All parameters (sample_mean, population_mean, standard_deviation, sample_size) are required."}), 400

        if sample_size <= 0:
            return jsonify({"error": "Sample size must be greater than zero."}), 400

        if standard_deviation <= 0:
            return jsonify({"error": "Standard deviation must be greater than zero."}), 400

        # Calculate the Z-score
        z_score = (sample_mean - population_mean) / (standard_deviation / math.sqrt(sample_size))

        # Calculate the p-value (two-tailed test)
        p_value = 2 * (1 - norm.cdf(abs(z_score)))

        # Generate bell curve data for graph
        x_values = [round(z, 2) for z in [i * 0.1 for i in range(-50, 51)]]  # Z-scores from -5 to 5
        y_values = [norm.pdf(z) for z in x_values]  # Probability density values

        # Return the results as JSON
        return jsonify({
            "z_score": z_score,
            "p_value": p_value,
            "conclusion": "Reject null hypothesis" if p_value < 0.05 else "Fail to reject null hypothesis",
            "plot_data": {
                "x_values": x_values,
                "y_values": y_values
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ztest_two_sample', methods=['POST'])
def two_sample_z_test():
    try:
        # Extracting data from the request
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided."}), 400

        # Retrieve parameters from the input
        sample_mean1 = data.get("sample_mean1")
        sample_mean2 = data.get("sample_mean2")
        std_dev1 = data.get("std_dev1")
        std_dev2 = data.get("std_dev2")
        n1 = data.get("sample_size1")
        n2 = data.get("sample_size2")

        # Validate inputs
        if None in [sample_mean1, sample_mean2, std_dev1, std_dev2, n1, n2]:
            return jsonify({"error": "All parameters (sample_mean1, sample_mean2, std_dev1, std_dev2, sample_size1, sample_size2) are required."}), 400

        if n1 <= 0 or n2 <= 0:
            return jsonify({"error": "Sample sizes must be greater than zero."}), 400

        if std_dev1 <= 0 or std_dev2 <= 0:
            return jsonify({"error": "Standard deviations must be greater than zero."}), 400

        # Calculate the Z-score for two samples
        pooled_std_error = math.sqrt((std_dev1**2 / n1) + (std_dev2**2 / n2))
        z_score = (sample_mean1 - sample_mean2) / pooled_std_error

        # Calculate the p-value (two-tailed test)
        p_value = 2 * (1 - norm.cdf(abs(z_score)))

        # Return the results as JSON
        return jsonify({
            "z_score": z_score,
            "p_value": p_value,
            "conclusion": "Reject null hypothesis" if p_value < 0.05 else "Fail to reject null hypothesis"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
