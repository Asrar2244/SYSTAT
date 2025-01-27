from flask import Blueprint, request, jsonify
from scipy.stats import t
import math

# Blueprint setup
t_test_api = Blueprint("t_test_api", __name__)

# ---------------------------------------------
# Helper Functions
# ---------------------------------------------
def validate_inputs(data, required_fields):
    """
    Validate that all required fields are present and non-null in the input data.
    :param data: Dictionary containing input data
    :param required_fields: List of required field names
    :return: (bool, str) - Validation status and error message (if any)
    """
    for field in required_fields:
        if field not in data or data[field] is None:
            return False, f"Missing required parameter: {field}."
    return True, ""

def calculate_confidence_interval(sample_mean, sample_std, sample_size, confidence_level, df):
    """
    Calculate the confidence interval for the t-test.
    :param sample_mean: Sample mean
    :param sample_std: Sample standard deviation
    :param sample_size: Sample size
    :param confidence_level: Confidence level (e.g., 0.95)
    :param df: Degrees of freedom
    :return: (lower_bound, upper_bound) of the confidence interval
    """
    critical_value = t.ppf((1 + confidence_level) / 2, df)
    margin_of_error = critical_value * (sample_std / math.sqrt(sample_size))
    return sample_mean - margin_of_error, sample_mean + margin_of_error

def generate_t_distribution_data(df, x_range=(-5, 5), step=0.1):
    """
    Generate x and y values for the t-distribution graph.
    :param df: Degrees of freedom
    :param x_range: Range of x values (default: -5 to 5)
    :param step: Step size for x values (default: 0.1)
    :return: x_values, y_values
    """
    x_values = [round(x, 2) for x in [i * step for i in range(int(x_range[0] / step), int(x_range[1] / step) + 1)]]
    y_values = [t.pdf(x, df) for x in x_values]
    return x_values, y_values

# ---------------------------------------------
# API Route
# ---------------------------------------------
@t_test_api.route('/one_sample_ttest', methods=['POST'])
def one_sample_t_test():
    try:
        # Extract data from request
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided."}), 400

        # Required parameters
        required_fields = ["sample_mean", "population_mean", "sample_std", "sample_size"]
        is_valid, error_message = validate_inputs(data, required_fields)
        if not is_valid:
            return jsonify({"error": error_message}), 400

        # Retrieve parameters with defaults
        dataset_id = data.get("dataset_id", "default_dataset")
        sample_mean = data["sample_mean"]
        population_mean = data["population_mean"]
        sample_std = data["sample_std"]
        sample_size = data["sample_size"]
        confidence_level = data.get("confidence_level", 0.95)
        alternative = data.get("alternative", "not_equal")  # "greater", "less", or "not_equal"

        # Validate numeric inputs
        if sample_size <= 1:
            return jsonify({"error": "Sample size must be greater than one for a t-test."}), 400
        if sample_std <= 0:
            return jsonify({"error": "Sample standard deviation must be greater than zero."}), 400
        if not (0 < confidence_level < 1):
            return jsonify({"error": "Confidence level must be between 0 and 1 (exclusive)."}), 400

        # Calculate t-score
        t_score = (sample_mean - population_mean) / (sample_std / math.sqrt(sample_size))

        # Degrees of freedom
        df = sample_size - 1

        # Calculate p-value based on the alternative hypothesis
        if alternative == "greater":
            p_value = 1 - t.cdf(t_score, df)
        elif alternative == "less":
            p_value = t.cdf(t_score, df)
        elif alternative == "not_equal":
            p_value = 2 * (1 - t.cdf(abs(t_score), df))
        else:
            return jsonify({"error": f"Invalid alternative hypothesis: {alternative}. Use 'greater', 'less', or 'not_equal'."}), 400

        # Determine conclusion
        alpha = 1 - confidence_level
        conclusion = "Reject null hypothesis" if p_value < alpha else "Fail to reject null hypothesis"

        # Calculate confidence interval
        confidence_interval_lower, confidence_interval_upper = calculate_confidence_interval(
            sample_mean, sample_std, sample_size, confidence_level, df
        )

        # Generate t-distribution data for graph
        x_values, y_values = generate_t_distribution_data(df)

        # Return the results as JSON
        return jsonify({
            "dataset_id": dataset_id,
            "t_score": t_score,
            "p_value": p_value,
            "degrees_of_freedom": df,
            "confidence_interval": {
                "lower": confidence_interval_lower,
                "upper": confidence_interval_upper,
                "confidence_level": confidence_level
            },
            "conclusion": conclusion,
            "plot_data": {
                "x_values": x_values,
                "y_values": y_values
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
