import streamlit as st
import pickle
import logging
import os
import sklearn

# Specify the version used to save the model, if known
 # replace with the actual version
required_sklearn_version = '1.2.2'
logging.basicConfig(level=logging.INFO)

def load_model(file_path):
    logging.info(f"Current scikit-learn version: {sklearn.__version__}")
    if sklearn.__version__ != required_sklearn_version:
        st.warning(f"Expected scikit-learn version {required_sklearn_version} but got {sklearn.__version__}")
    
    if not os.path.exists(file_path):
        st.error(f"Model file not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError as fnf_error:
        logging.error(f"File not found: {fnf_error}")
        st.error(f"File not found: {fnf_error}")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        st.error(f"Error loading model: {e}")
    return None

def main():
    st.title("Ensemble Model Inference with Streamlit")
    
    model = load_model('Ensemble (3).sav')
    if model is None:
        return
    
    # Example of using the loaded model
    st.write("Model loaded successfully!")
    # Add further code to use the model for predictions, etc.

if __name__ == "__main__":
    main()
