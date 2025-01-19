import math
from scipy.stats import norm
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

@app.route('/two_sample_z_test', methods=['POST'])
def two_sample_z_test():
    try:
        # Retrieve input from the frontend
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided."}), 400

        # Required keys
        required_keys = ['mean1', 'mean2', 'std1', 'std2', 'n1', 'n2']
        missing_keys = [key for key in required_keys if key not in data or data[key] is None]

        if missing_keys:
            return jsonify({"error": f"Missing or invalid parameters: {', '.join(missing_keys)}"}), 400

        # Extract group statistics
        mean1 = float(data['mean1'])
        mean2 = float(data['mean2'])
        std1 = float(data['std1'])
        std2 = float(data['std2'])
        n1 = int(data['n1'])
        n2 = int(data['n2'])

        if n1 <= 0 or n2 <= 0 or std1 <= 0 or std2 <= 0:
            return jsonify({"error": "Sample sizes and standard deviations must be greater than zero."}), 400

        # Labels for groups
        group1_label = data.get("group1_label", "Group 1")
        group2_label = data.get("group2_label", "Group 2")

        # Perform Two-Sample Z-Test
        pooled_std = math.sqrt((std1 ** 2 / n1) + (std2 ** 2 / n2))
        z_score = (mean1 - mean2) / pooled_std
        p_value = 2 * (1 - norm.cdf(abs(z_score)))

        # Confidence interval
        mean_diff = mean1 - mean2
        ci_low = mean_diff - 1.96 * pooled_std
        ci_high = mean_diff + 1.96 * pooled_std

        # Prepare response data for plotting
        plot_data = {
            "z_score": z_score,
            "p_value": p_value,
            "mean_difference": mean_diff,
            "confidence_interval": {
                "lower_bound": ci_low,
                "upper_bound": ci_high
            },
            "group_statistics": {
                "group1": {
                    "label": group1_label,
                    "mean": mean1,
                    "std": std1,
                    "size": n1
                },
                "group2": {
                    "label": group2_label,
                    "mean": mean2,
                    "std": std2,
                    "size": n2
                }
            }
        }

        # Return the results as JSON
        return jsonify(plot_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
