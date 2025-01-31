import os
from flask import Flask


def create_app():
    app = Flask(__name__)

  

    # Import blueprint inside create_app to avoid circular imports
    from app.api.z_test_api import z_test_api
    app.register_blueprint(z_test_api, url_prefix='/ztest')

    # Import and register two_sample_z_test blueprint
    from app.api.two_sample_z_test_api import two_sample_z_test
    app.register_blueprint(two_sample_z_test, url_prefix='/api')

    # Import inside function to avoid circular import
    from app.api.One_sample_t_test_api import t_test_api
    app.register_blueprint(t_test_api, url_prefix='/ttest')

    # Import blueprint inside create_app to avoid circular imports
    from app.api.paired_t_test_api import paired_t_test_api
    app.register_blueprint(paired_t_test_api, url_prefix='/api')

       # Import and register the two-sample t-test blueprint
    from app.api.two_sample_t_test_api import two_sample_t_test_api
    app.register_blueprint(two_sample_t_test_api, url_prefix='/api')

    return app
