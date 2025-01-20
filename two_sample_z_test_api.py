import math
from scipy.stats import norm
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Database initialization (for demonstration purposes)
def init_db():
    conn = sqlite3.connect("test_results.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS test_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT NOT NULL,
            mean1 REAL NOT NULL,
            mean2 REAL NOT NULL,
            std1 REAL NOT NULL,
            std2 REAL NOT NULL,
            n1 INTEGER NOT NULL,
            n2 INTEGER NOT NULL,
            z_score REAL NOT NULL,
            p_value REAL NOT NULL,
            mean_difference REAL NOT NULL,
            ci_lower REAL NOT NULL,
            ci_upper REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

@app.route('/two_sample_z_test', methods=['POST'])
def two_sample_z_test():
    try:
        # Retrieve input from the frontend
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided."}), 400

        # Required keys
        required_keys = ['dataset_id', 'mean1', 'mean2', 'std1', 'std2', 'n1', 'n2']
        missing_keys = [key for key in required_keys if key not in data or data[key] is None]

        if missing_keys:
            return jsonify({"error": f"Missing or invalid parameters: {', '.join(missing_keys)}"}), 400

        # Extract group statistics
        dataset_id = data['dataset_id']
        mean1 = float(data['mean1'])
        mean2 = float(data['mean2'])
        std1 = float(data['std1'])
        std2 = float(data['std2'])
        n1 = int(data['n1'])
        n2 = int(data['n2'])

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

        # Store the results in the database
        conn = sqlite3.connect("test_results.db")
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO test_results (dataset_id, mean1, mean2, std1, std2, n1, n2, z_score, p_value, mean_difference, ci_lower, ci_upper)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (dataset_id, mean1, mean2, std1, std2, n1, n2, z_score, p_value, mean_diff, ci_low, ci_high)
        )
        conn.commit()
        conn.close()

        # Prepare response data
        response_data = {
            "dataset_id": dataset_id,
            "z_score": z_score,
            "p_value": p_value,
            "mean_difference": mean_diff,
            "confidence_interval": {
                "lower_bound": ci_low,
                "upper_bound": ci_high
            }
        }

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
