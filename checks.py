import joblib
import streamlit as st
import sklearn

# Function to check the scikit-learn version
def check_sklearn_version(expected_version="1.2.2"):
    version = sklearn.__version__
    st.write(f"scikit-learn version: {version}")
    if version != expected_version:
        st.error(f"Expected scikit-learn version {expected_version} but got {version}. Please install the correct version.")
        return False
    return True

# Function to load the model
def load_model(file_path):
    try:
        model = joblib.load(file_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    st.title("Model Loader with scikit-learn Version Check")
    
    # Check scikit-learn version
    if not check_sklearn_version():
        return
    
    # Load model
    model_path = 'Ensemble (3).sav'
    model = load_model(model_path)
    
    if model:
        st.write("Model loaded successfully!")
        # Example of using the loaded model (add your code here)
    else:
        st.write(f"Failed to load the model from {model_path}. Ensure compatibility with scikit-learn version 1.2.2")

if __name__ == "__main__":
    main()
