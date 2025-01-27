# run.py

from app import create_app  # Import the create_app function from the app package

def main():
    # Create the Flask application instance
    app = create_app()

    # Run the application
    app.run(debug=True)

if __name__ == "__main__":
    main()
