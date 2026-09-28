import streamlit as st
import pandas as pd
import joblib


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👨‍💼",
    layout="wide"
)


# ==========================================
# LOAD DATA AND MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("employee_attrition_xgboost.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("data/employee_attrition.csv")


model = load_model()
df = load_data()


# ==========================================
# TITLE
# ==========================================

st.title("👨‍💼 Employee Attrition Prediction System")

st.write(
    """
    This system uses Machine Learning to predict whether an employee
    is likely to leave the organization based on employee-related factors.
    """
)

st.divider()


# ==========================================
# REMOVE UNNECESSARY COLUMNS
# ==========================================

features_to_remove = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours",
    "Attrition"
]

feature_df = df.drop(
    columns=[
        col for col in features_to_remove
        if col in df.columns
    ]
)


# ==========================================
# INPUT FORM
# ==========================================

st.subheader("📋 Enter Employee Details")

st.info(
    "Enter the employee information below and click "
    "'Predict Attrition'."
)


input_data = {}


with st.form("employee_form"):

    columns = feature_df.columns.tolist()

    # Divide fields into two columns
    col1, col2 = st.columns(2)

    for i, column in enumerate(columns):

        current_col = col1 if i % 2 == 0 else col2

        with current_col:

            # Categorical columns
            if feature_df[column].dtype == "object":

                options = sorted(
                    feature_df[column].dropna().unique().tolist()
                )

                input_data[column] = st.selectbox(
                    column,
                    options
                )

            # Numerical columns
            else:

                min_value = float(
                    feature_df[column].min()
                )

                max_value = float(
                    feature_df[column].max()
                )

                median_value = float(
                    feature_df[column].median()
                )

                input_data[column] = st.number_input(
                    column,
                    min_value=min_value,
                    max_value=max_value,
                    value=median_value
                )


    submitted = st.form_submit_button(
        "🔮 Predict Attrition"
    )


# ==========================================
# PREDICTION
# ==========================================

if submitted:

    # Convert input into dataframe
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Get probability
    probability = model.predict_proba(
        input_df
    )[0][1]

    risk_percentage = probability * 100


    st.divider()

    st.subheader("📊 Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ HIGH RISK — Employee may leave the organization."
        )

    else:

        st.success(
            "✅ LOW RISK — Employee is likely to stay."
        )


    # Display probability
    st.metric(
        "Predicted Attrition Risk",
        f"{risk_percentage:.2f}%"
    )


    # Risk interpretation
    if risk_percentage >= 70:

        st.warning(
            "Very High Attrition Risk"
        )

    elif risk_percentage >= 50:

        st.warning(
            "High Attrition Risk"
        )

    elif risk_percentage >= 30:

        st.info(
            "Moderate Attrition Risk"
        )

    else:

        st.success(
            "Low Attrition Risk"
        )


    # Probability bar
    st.progress(
        min(int(risk_percentage), 100)
    )


    st.subheader("💡 Interpretation")

    if prediction == 1:

        st.write(
            """
            The model predicts that this employee has a higher
            probability of leaving the organization. HR teams can
            examine factors such as overtime, job satisfaction,
            income, business travel and career progression.
            """
        )

    else:

        st.write(
            """
            The model predicts that this employee is more likely
            to remain with the organization based on the provided
            characteristics.
            """
        )