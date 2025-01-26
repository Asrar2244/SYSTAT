from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.stats import t
import math
import sqlite3

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# SQLite Database setup
DATABASE = "hypothesis_testing.db"


def init_db():
    """Initialize or update the database schema."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS t_test_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT UNIQUE,
            sample_mean REAL,
            population_mean REAL,
            sample_std REAL,
            sample_size INTEGER,
            t_score REAL,
            p_value REAL,
            confidence_interval_lower REAL,
            confidence_interval_upper REAL,
            conclusion TEXT
        )
    ''')
    conn.commit()
    conn.close()


@app.route('/ttest', methods=['POST'])
def one_sample_t_test():
    try:
        # Extracting data from the request
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided."}), 400

        # Retrieve parameters
        dataset_id = data.get("dataset_id", "default_dataset")
        sample_mean = data.get("sample_mean")
        population_mean = data.get("population_mean")
        sample_std = data.get("sample_std")
        sample_size = data.get("sample_size")
        confidence_level = data.get("confidence_level", 0.95)  # Default to 95%
        alternative = data.get("alternative", "not_equal")  # Default: two-tailed
        adjustment = data.get("adjustment")  # Optional: "bonferroni" or "dunn_sidak"
        num_comparisons = data.get("num_comparisons", 1)  # Default: single comparison

        # Validate inputs
        if None in [sample_mean, population_mean, sample_std, sample_size]:
            return jsonify({"error": "Missing required parameters."}), 400

        if sample_size <= 1:
            return jsonify({"error": "Sample size must be greater than one for a t-test."}), 400

        if sample_std <= 0:
            return jsonify({"error": "Sample standard deviation must be greater than zero."}), 400

        if not (0 < confidence_level < 1):
            return jsonify({"error": "Confidence level must be between 0 and 1 (exclusive)."}), 400

        # Calculate the t-score
        t_score = (sample_mean - population_mean) / (sample_std / math.sqrt(sample_size))

        # Degrees of freedom
        df = sample_size - 1

        # Calculate p-value based on alternative hypothesis
        if alternative == "greater":
            p_value = 1 - t.cdf(t_score, df)
        elif alternative == "less":
            p_value = t.cdf(t_score, df)
        else:  # "not_equal" (two-tailed test)
            p_value = 2 * (1 - t.cdf(abs(t_score), df))

        # Apply adjustments if specified
        if adjustment == "bonferroni" and num_comparisons > 1:
            p_value = min(p_value * num_comparisons, 1)
        elif adjustment == "dunn_sidak" and num_comparisons > 1:
            p_value = min(1 - (1 - p_value) ** num_comparisons, 1)

        # Determine conclusion
        alpha = 1 - confidence_level
        conclusion = "Reject null hypothesis" if p_value < alpha else "Fail to reject null hypothesis"

        # Calculate confidence interval
        critical_value = t.ppf((1 + confidence_level) / 2, df)
        margin_of_error = critical_value * (sample_std / math.sqrt(sample_size))
        confidence_interval_lower = sample_mean - margin_of_error
        confidence_interval_upper = sample_mean + margin_of_error

        # Update database with results
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO t_test_data (
                dataset_id, sample_mean, population_mean, sample_std,
                sample_size, t_score, p_value, confidence_interval_lower,
                confidence_interval_upper, conclusion
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            dataset_id, sample_mean, population_mean, sample_std, sample_size,
            t_score, p_value, confidence_interval_lower, confidence_interval_upper,
            conclusion
        ))
        conn.commit()
        conn.close()

        # Generate t-distribution data for graph
        x_values = [round(x, 2) for x in [i * 0.1 for i in range(-50, 51)]]  # t-scores from -5 to 5
        y_values = [t.pdf(x, df) for x in x_values]  # Probability density values

        # Return the results as JSON
        return jsonify({
            "t_score": t_score,
            "p_value": p_value,
            "degrees_of_freedom": df,
            "confidence_interval": {
                "lower": confidence_interval_lower,
                "upper": confidence_interval_upper,
                "confidence_level": confidence_level
            },
            "conclusion": conclusion,
            "adjustment": adjustment,
            "num_comparisons": num_comparisons,
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
