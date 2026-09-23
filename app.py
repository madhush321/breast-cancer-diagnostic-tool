import streamlit as st
import pandas as pd
import joblib

# 1. Load your saved model (ensure 'model.pkl' is in the same folder)
classifier = joblib.load('classifier.pkl')

st.title("Breast Cancer Prediction Model")
st.write("Adjust the cell characteristics below (scale 1-10) to generate a prediction.")

# 2. Design the UI using columns for a cleaner layout
col1, col2, col3 = st.columns(3)

with col1:
    clump = st.slider("Clump Thickness", 1, 10, 5)
    cell_size = st.slider("Uniformity of Cell Size", 1, 10, 5)
    cell_shape = st.slider("Uniformity of Cell Shape", 1, 10, 5)

with col2:
    adhesion = st.slider("Marginal Adhesion", 1, 10, 5)
    epithelial = st.slider("Single Epithelial Cell Size", 1, 10, 5)
    bare_nuclei = st.slider("Bare Nuclei", 1, 10, 5)

with col3:
    chromatin = st.slider("Bland Chromatin", 1, 10, 5)
    nucleoli = st.slider("Normal Nucleoli", 1, 10, 5)
    mitoses = st.slider("Mitoses", 1, 10, 5)

# 3. Create the prediction logic
if st.button("Predict Classification", type="primary"):
    
    # Map the UI inputs to a DataFrame with your EXACT column names
    user_data = pd.DataFrame({
        'Clump_Thickness': [clump],
        'Uniformity_of_Cell_Size': [cell_size],
        'Uniformity_of_Cell_Shape': [cell_shape],
        'Marginal_Adhesion': [adhesion],
        'Single_Epithelial_Cell_Size': [epithelial],
        'Bare_Nuclei': [bare_nuclei],
        'Bland_Chromatin': [chromatin],
        'Normal_Nucleoli': [nucleoli],
        'Mitoses': [mitoses]
    })

    # Generate the prediction
    prediction = classifier.predict(user_data)
    result = prediction[0]

    # Display results (Adjusting for common target variable formats: 0/1 or 2/4)
    st.divider()
    if result == 4 or result == 1:
        st.error("Prediction:  Malignant")
        st.write("means a tumor is cancerous. The cells grow uncontrollably, can invade surrounding tissues, and have the potential to break away and spread to other parts of the body (metastasize).")
        st.write("Please consult with a medical professional for further analysis.")
    elif result == 2 or result == 0:
        st.success("Prediction:  Benign")
    else:
        st.info(f"Model Output Class: {result}")