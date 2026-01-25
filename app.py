import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# Load trained models
diabetes_model = pickle.load(open("models/diabetes_model.pkl", "rb"))
heart_model = pickle.load(open("models/heart_model.pkl", "rb"))
brain_model = pickle.load(open("models/brainstroke_model.pkl", "rb"))

st.set_page_config(page_title="Disease Prediction System", layout="wide")

# Session state for uploaded dataset
if "df" not in st.session_state:
    st.session_state.df = None

# ---------------- Sidebar ----------------
st.sidebar.title("🩺 Disease Prediction System")
menu = st.sidebar.radio("Navigation", ["Home", "Predict Disease", "Upload Dataset", "Update Dataset", "Data Visualizations"])

# ---------------- Home ----------------
if menu == "Home":
    st.title("Disease Prediction System using Machine Learning")
    st.write("""
    This application predicts:
    - **Diabetes**
    - **Heart Disease**
    - **Brain Stroke**

    It also allows users to upload any dataset, explore it, preprocess it, and visualize it interactively.
    """)

# ---------------- Disease Prediction ----------------
elif menu == "Predict Disease":
    disease = st.sidebar.selectbox("Select Disease", ["Diabetes", "Heart Disease", "Brain Stroke"])

    # -------- DIABETES --------
    if disease == "Diabetes":
        st.title("Diabetes Prediction")

        pregnancies = st.number_input("Pregnancies", 0, 20)
        glucose = st.number_input("Glucose Level", 50, 200)
        blood_pressure = st.number_input("Blood Pressure", 30, 150)
        skin_thickness = st.number_input("Skin Thickness", 0, 100)
        insulin = st.number_input("Insulin Level", 0, 900)
        bmi = st.number_input("BMI", 10.0, 70.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
        age = st.number_input("Age", 10, 100)

        if st.button("Predict Diabetes"):
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                     insulin, bmi, dpf, age]])
            prediction = diabetes_model.predict(input_data)[0]
            st.success("🟢 Not Diabetic" if prediction == 0 else "🔴 Diabetic")

    # -------- HEART DISEASE --------
    elif disease == "Heart Disease":
        st.title("Heart Disease Prediction")

        age = st.number_input("Age", 20, 100)
        sex = st.selectbox("Sex", ["Male", "Female"])
        sex_val = 1 if sex == "Male" else 0

        cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"])
        cp_val = ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"].index(cp)

        trestbps = st.number_input("Resting Blood Pressure", 80, 200)
        chol = st.number_input("Cholesterol", 100, 600)

        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        fbs_val = 1 if fbs == "Yes" else 0

        restecg = st.selectbox("Resting ECG", ["Normal", "ST-T abnormality", "Left Ventricular Hypertrophy"])
        restecg_val = ["Normal", "ST-T abnormality", "Left Ventricular Hypertrophy"].index(restecg)

        thalach = st.number_input("Maximum Heart Rate", 60, 220)

        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        exang_val = 1 if exang == "Yes" else 0

        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.0)

        slope = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])
        slope_val = ["Upsloping", "Flat", "Downsloping"].index(slope)

        ca = st.number_input("Number of Major Vessels", 0, 4)

        thal = st.selectbox("Thalassemia", ["Normal", "Fixed defect", "Reversible defect"])
        thal_val = ["Normal", "Fixed defect", "Reversible defect"].index(thal) + 1

        if st.button("Predict Heart Disease"):
            input_data = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val,
                                     restecg_val, thalach, exang_val, oldpeak,
                                     slope_val, ca, thal_val]])
            prediction = heart_model.predict(input_data)[0]
            st.success("🟢 No Heart Disease" if prediction == 0 else "🔴 Heart Disease Detected")

    # -------- BRAIN STROKE --------
    elif disease == "Brain Stroke":
        st.title("Brain Stroke Prediction")

        gender = st.selectbox("Gender", ["Male", "Female"])
        gender_val = 1 if gender == "Male" else 0

        age = st.number_input("Age", 10, 100)

        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        hypertension_val = 1 if hypertension == "Yes" else 0

        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        heart_disease_val = 1 if heart_disease == "Yes" else 0

        ever_married = st.selectbox("Ever Married", ["No", "Yes"])
        ever_married_val = 1 if ever_married == "Yes" else 0

        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        work_type_val = ["Private", "Self-employed", "Govt_job", "children", "Never_worked"].index(work_type)

        residence = st.selectbox("Residence Type", ["Urban", "Rural"])
        residence_val = 1 if residence == "Urban" else 0

        avg_glucose = st.number_input("Average Glucose Level", 50.0, 300.0)
        bmi = st.number_input("BMI", 10.0, 70.0)

        smoking = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])
        smoking_val = ["formerly smoked", "never smoked", "smokes", "Unknown"].index(smoking)

        if st.button("Predict Brain Stroke"):
            input_data = np.array([[gender_val, age, hypertension_val, heart_disease_val,
                                     ever_married_val, work_type_val, residence_val,
                                     avg_glucose, bmi, smoking_val]])
            prediction = brain_model.predict(input_data)[0]
            st.success("🟢 No Brain Stroke" if prediction == 0 else "🔴 Brain Stroke Detected")

# ---------------- Upload Dataset ----------------
elif menu == "Upload Dataset":
    st.title("Upload a Dataset")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("Dataset uploaded successfully!")
        st.write(df.head())

        st.subheader("Answer the following based on your dataset")
        inputs = []

        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.number):
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                val = st.number_input(f"{col}", min_value=min_val, max_value=max_val)
                inputs.append(val)
            else:
                options = df[col].astype(str).unique().tolist()
                val = st.selectbox(f"{col}", options)
                inputs.append(val)

        if st.button("Submit Inputs"):
            st.write("Inputs captured successfully.")
            st.info("Model prediction for uploaded dataset is dataset-dependent and must be connected manually.")

# ---------------- Update Dataset ----------------
elif menu == "Update Dataset":
    st.title("Update or Modify Uploaded Dataset")

    if st.session_state.df is not None:
        df = st.session_state.df

        # Drop Columns
        st.subheader("Drop Columns")
        drop_cols = st.multiselect("Select columns to drop", df.columns)
        if st.button("Drop Selected Columns"):
            df.drop(columns=drop_cols, inplace=True)
            st.success(f"Dropped columns: {drop_cols}")
            st.write(df.head())

        # Fill Missing Values
        st.subheader("Fill Missing Values")
        col_fill = st.selectbox("Select column", df.columns)
        method = st.radio("Fill with", ["Mean", "Median", "Mode"])
        if st.button("Fill Missing Values"):
            if method == "Mean":
                value = df[col_fill].mean()
            elif method == "Median":
                value = df[col_fill].median()
            else:
                value = df[col_fill].mode()[0]
            df[col_fill].fillna(value, inplace=True)
            st.success(f"Filled missing values in {col_fill} using {method}")
            st.write(df.head())

        # Encode Categorical Columns
        st.subheader("Encode Categorical Columns")
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        if cat_cols:
            encode_col = st.selectbox("Select column to encode", cat_cols)
            if st.button("Encode Column"):
                le = LabelEncoder()
                df[encode_col] = le.fit_transform(df[encode_col])
                st.success(f"Encoded column: {encode_col}")
                st.write(df.head())
        else:
            st.info("No categorical columns to encode.")

        # Scale Numerical Columns
        st.subheader("Scale Numerical Columns")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        scaler_type = st.radio("Select scaler", ["MinMaxScaler", "StandardScaler"])
        if st.button("Scale Numerical Columns"):
            scaler = MinMaxScaler() if scaler_type == "MinMaxScaler" else StandardScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])
            st.success(f"Scaled numerical columns using {scaler_type}")
            st.write(df.head())

        st.session_state.df = df
    else:
        st.warning("Please upload a dataset first.")

# ---------------- Data Visualizations ----------------
elif menu == "Data Visualizations":
    st.title("Dataset Visualizations")

    if st.session_state.df is not None:
        df = st.session_state.df

        col = st.selectbox("Select column to visualize", df.columns)

        if np.issubdtype(df[col].dtype, np.number):
            st.subheader(f"Histogram of {col}")
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)

            st.subheader(f"Boxplot of {col}")
            fig2, ax2 = plt.subplots()
            sns.boxplot(y=df[col], ax=ax2)
            st.pyplot(fig2)

        else:
            st.subheader(f"Countplot of {col}")
            fig3, ax3 = plt.subplots()
            sns.countplot(x=df[col], ax=ax3)
            st.pyplot(fig3)

        st.subheader("Correlation Heatmap")
        fig4, ax4 = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax4)
        st.pyplot(fig4)

    else:
        st.warning("Please upload a dataset first.")
