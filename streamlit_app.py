import streamlit as st
import pandas as pd
import pickle
import joblib

# Function to load the model
@st.cache_data
def load_model():
    with open('kidney_disease_model', 'rb') as file:
        loaded_model = joblib.load(file)
    return loaded_model

# Load your model
loaded_model = load_model()

# Function to create the input datafram
def create_input_df(user_inputs, category_map):
    # Transform categorical inputs using category_map
    for category in category_map:
        if category in user_inputs:
            user_inputs[category] = category_map[category].get(user_inputs[category], -1)  # -1 or other value for missing/unknown categories
    input_df = pd.DataFrame([user_inputs])
    return input_df

# Define your category_map
category_map = {
    'red_blood_cells': {'normal': 0, 'abnormal': 1},
    'pus_cell': {'normal': 0, 'abnormal': 1},
    'pus_cell_clumps': {'notpresent': 0, 'present': 1},
    'bacteria': {'notpresent': 0, 'present': 1},
    'hypertension': {'no': 0, 'yes': 1},
    'diabetes_mellitus': {'no': 0, 'yes': 1},
    'coronary_artery_disease': {'no': 0, 'yes': 1},
    'appetite': {'poor': 0, 'good': 1},
    'pedal_edema': {'no': 0, 'yes': 1},
    'anemia': {'no': 0, 'yes': 1}

}
# Sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.selectbox('Select a page:', 
                           ['Prediction', 'Code', 'About'])

if options == 'Prediction': # Prediction page
    st.title('Chronic Kidney Disease Prediction')

    # categorical input
    red_blood_cells = st.radio('Red Blood Cells', ('normal', 'abnormal'))
    pus_cell = st.radio('Pus Cell', ('normal', 'abnormal'))
    pus_cell_clumps = st.radio('Pus Cell Clumps', ('notpresent', 'present'))
    bacteria = st.radio('Bacteria', ('notpresent', 'present'))
    hypertension = st.radio('Hypertension', ('no', 'yes'))
    diabetes_mellitus = st.radio('Diabetes Mellitus', ('no', 'yes'))
    coronary_artery_disease = st.radio('Coronary Artery Disease', ('no', 'yes'))
    appetite = st.radio('Appetite', ('poor', 'good'))
    pedal_edema = st.radio('Pedal Edema', ('no', 'yes'))
    anemia = st.radio('Anemia', ('no', 'yes'))

    # numerical input
    age = st.slider('Age', 0, 100, 0)
    blood_pressure = st.slider('Blood Pressure', 0, 180, 0)
    specific_gravity = st.slider('Specific Gravity', 0.0, 2.0, 0.0)
    albumin = st.slider('Albumin', 0, 5, 0)
    sugar = st.slider('Sugar', 0, 5, 0)
    blood_glucose_random = st.slider('Blood Glucose Random', 0, 500, 0)
    blood_urea = st.slider('Blood Urea', 0, 200, 0)
    serum_creatinine = st.slider('Serum Creatinine', 0.0, 10.0, 0.0)
    sodium = st.slider('Sodium', 0, 200, 0)
    potassium = st.slider('Potassium', 0, 10, 0)
    hemoglobin = st.slider('Hemoglobin', 0, 20, 0)
    packed_cell_volume = st.slider('Packed Cell Volume', 0, 100, 0)
    white_blood_cell_count = st.slider('White Blood Cell Count', 0, 20000, 0)
    red_blood_cell_count = st.slider('Red Blood Cell Count', 0, 10, 0)

    # User inputs
    user_inputs = {
        'age': age, 
        'blood_pressure': blood_pressure, 
        'specific_gravity': specific_gravity, 
        'albumin': albumin, 
        'sugar': sugar,
        'red_blood_cells': red_blood_cells, 
        'pus_cell': pus_cell, 
        'pus_cell_clumps': pus_cell_clumps, 
        'bacteria': bacteria,
        'blood_glucose_random': blood_glucose_random, 
        'blood_urea':blood_urea, 
        'serum_creatinine': serum_creatinine, 
        'sodium': sodium,
        'potassium': potassium, 
        'hemoglobin': hemoglobin, 
        'packed_cell_volume': packed_cell_volume,
        'white_blood_cell_count': white_blood_cell_count, 
        'red_blood_cell_count': red_blood_cell_count, 
        'hypertension': hypertension,
        'diabetes_mellitus': diabetes_mellitus, 
        'coronary_artery_disease': coronary_artery_disease, 
        'appetite': appetite,
        'pedal_edema': pedal_edema, 
        'anemia': anemia
        }

    # Create a button to predict the output
    if st.button('Predict'):
        input_df = create_input_df(user_inputs, category_map)
        prediction = loaded_model.predict(input_df)
        # If 0 : Chronic Kidney Disease present
        # If 1 : Chronic Kidney Disease not present
        if prediction[0] == 0:
            st.write('The patient is likely to have Chronic Kidney Disease.')
        else:
            st.write('The patient is likely to not have Chronic Kidney Disease.')
        st.write('--'*50)
        
        with st.expander("Show more details"):
            st.write("Details of the prediction:")
            st.json(loaded_model.get_params())
            st.write('Model used: Support Vector Machine (SVM)')
            
elif options == 'Code':
    st.header('Code')
    # Add a button to download the Jupyter notebook (.ipynb) file
    notebook_path = 'model.ipynb'
    with open(notebook_path, "rb") as file:
        btn = st.download_button(
            label="Download Jupyter Notebook",
            data=file,
            file_name="Chronic Kidney Disease Prediction.ipynb",
            mime="application/x-ipynb+json"
        )
    st.write('You can download the Jupyter notebook to view the code and the model building process.')
    st.write('--'*50)

    st.header('Data')
    # Add a button to download your dataset
    data_path = 'kidney_disease.csv'
    with open(data_path, "rb") as file:
        btn = st.download_button(
            label="Download Dataset",
            data=file,
            file_name="kidney_disease.csv",
            mime="text/csv"
        )
    st.write('You can download the dataset to use it for your own analysis or model building.')
    st.write('--'*50)

    st.header('GitHub Repository')
    st.write('You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/gokulnpc/Chronic-Kidney-Disease-Prediction)')
    st.write('--'*50)

elif options == 'About':
    st.title('About')
    st.write('This we app is created to predict the chronic kidney disease using the data from the UCI Machine Learning Repository.')
    st.write('The dataset contains various parameters like age, blood pressure, specific gravity, albumin, sugar, red blood cells, pus cell, pus cell clumps, bacteria, blood glucose random, blood urea, serum creatinine, sodium, potassium, hemoglobin, packed cell volume, white blood cell count, red blood cell count, hypertension, diabetes mellitus, coronary artery disease, appetite, pedal edema, anemia, and class.')
    st.write('The dataset is used to build a machine learning model to predict the chronic kidney disease based on the input parameters.')
    st.write('--'*50)

    st.write('The web app is open-source. You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/gokulnpc/Chronic-Kidney-Disease-Prediction)')
    st.write('--'*50)

    st.header('Contact')
    st.write('You can contact me for any queries or feedback:')
    st.write('Email: gokulnpc@gmail.com')
    st.write('LinkedIn: [Gokuleshwaran Narayanan](https://www.linkedin.com/in/gokulnpc/)')
    st.write('--'*50)
