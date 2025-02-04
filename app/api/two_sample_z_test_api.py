import pandas as pd
from flask import Blueprint, request, jsonify
from statsmodels.stats.weightstats import ztest
from scipy.stats import norm
from app.utils import read_input_data
from app.logger import logger

# Initialize the Blueprint
two_sample_z_test = Blueprint('two_sample_z_test', __name__)

@two_sample_z_test.route('/', methods=['POST'])
def perform_two_sample_z_test():
    """Perform a two-sample Z-test and return statistical results."""
    try:
        logger.info("Received request for two-sample Z-test.")
        
        # Read input data
        if request.is_json:
            data = request.get_json()
        elif 'file' in request.files:
            file = request.files['file']
            data = read_input_data(file)
        else:
            logger.error("No data provided. JSON or file input required.")
            return jsonify({"error": "No data provided. Please provide either JSON or file input."}), 400
        
        # Extract parameters
        column = data.get('column')
        group_col = data.get('group_column')
        confidence = float(data.get('confidence', 0.95))
        alternative = data.get('alternative', 'two-sided').lower()
        
        logger.info(f"Parameters: column={column}, group_col={group_col}, confidence={confidence}, alternative={alternative}")
        
        # Convert data to DataFrame
        df = pd.DataFrame(data['data'])
        
        # Ensure there are exactly two groups
        groups = df[group_col].unique()
        if len(groups) != 2:
            logger.error("Invalid grouping variable. Ensure exactly two groups.")
            return jsonify({"error": "Invalid grouping variable. Ensure exactly two groups."}), 400
        
        # Separate the data into two groups
        group1_data = df[df[group_col] == groups[0]][column]
        group2_data = df[df[group_col] == groups[1]][column]
        
        # Perform two-sample Z-test
        z_stat, p_value = ztest(group1_data, group2_data, alternative=alternative)
        
        # Compute confidence interval
        mean_diff = group1_data.mean() - group2_data.mean()
        std_err = (group1_data.std()**2 / len(group1_data) + group2_data.std()**2 / len(group2_data))**0.5
        z_critical = norm.ppf(1 - (1 - confidence) / 2)
        ci_low = mean_diff - z_critical * std_err
        ci_high = mean_diff + z_critical * std_err
        
        # Conclusion
        conclusion = "Significant difference between the means." if p_value < (1 - confidence) else "No significant difference between the means."
        
        # Prepare results
        results = {
            "hypothesis": f"Ho: Mean1 = Mean2 vs H1: Mean1 {'!=' if alternative == 'two-sided' else ('<' if alternative == 'smaller' else '>')} Mean2",
            "grouping_variable": group_col,
            "summary": {
                f"{groups[0]}": {"N": len(group1_data), "Mean": group1_data.mean()},
                f"{groups[1]}": {"N": len(group2_data), "Mean": group2_data.mean()}
            },
            "confidence_interval": {
                "confidence_level": confidence,
                "mean_difference": mean_diff,
                "lower_bound": ci_low,
                "upper_bound": ci_high
            },
            "z_stat": z_stat,
            "p_value": p_value,
            "conclusion": conclusion
        }
        
        logger.info("Two-sample Z-test completed successfully.")
        return jsonify(results), 200
    
    except Exception as e:
        logger.error(f"Error during two-sample Z-test: {str(e)}")
        return jsonify({"error": str(e)}), 500
