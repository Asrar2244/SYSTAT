from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.stats import norm
import math
import sqlite3

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Connect to SQLite database
DATABASE = "hypothesis_testing.db"


def init_db():
    """Initialize the database with a test dataset table."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS test_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT UNIQUE,
            sample_mean REAL,
            population_mean REAL,
            standard_deviation REAL,
            sample_size INTEGER,
            z_score REAL,
            p_value REAL,
            conclusion TEXT
        )
    ''')
    conn.commit()
    conn.close()


@app.route('/ztest', methods=['POST'])
def one_sample_z_test():
    try:
        # Extracting data from the request
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided."}), 400

        # Retrieve dataset_id and other parameters
        dataset_id = data.get("dataset_id")
        if not dataset_id:
            return jsonify({"error": "Dataset ID is required."}), 400

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

        # Determine conclusion
        conclusion = "Reject null hypothesis" if p_value < 0.05 else "Fail to reject null hypothesis"

        # Update database with results
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO test_data (
                dataset_id, sample_mean, population_mean, standard_deviation,
                sample_size, z_score, p_value, conclusion
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (dataset_id, sample_mean, population_mean, standard_deviation, sample_size, z_score, p_value, conclusion))
        conn.commit()
        conn.close()

        # Generate bell curve data for graph
        x_values = [round(z, 2) for z in [i * 0.1 for i in range(-50, 51)]]  # Z-scores from -5 to 5
        y_values = [norm.pdf(z) for z in x_values]  # Probability density values

        # Return the results as JSON
        return jsonify({
            "z_score": z_score,
            "p_value": p_value,
            "conclusion": conclusion,
            "plot_data": {
                "x_values": x_values,
                "y_values": y_values
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    init_db()  # Initialize the database when the app starts
    app.run(debug=True)
