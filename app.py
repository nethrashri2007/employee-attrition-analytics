import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Employee Attrition Analytics",
    page_icon="👥",
    layout="wide"
)


# =========================================================
# LOAD DATA AND MODEL
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/employee_attrition.csv")


@st.cache_resource
def load_model():
    return joblib.load("employee_attrition_xgboost.pkl")


df = load_data()
model = load_model()


# =========================================================
# BASIC PREPARATION
# =========================================================

# Convert Attrition into numeric form only for calculations
df["Attrition_Flag"] = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("👥 Employee Attrition")

st.sidebar.write(
    "Navigate through the employee analytics dashboard."
)

page = st.sidebar.radio(
    "Select Section",
    [
        "📊 Dashboard",
        "🔍 Employee Analysis",
        "🤖 Attrition Prediction",
        "📈 Model Performance"
    ]
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "📊 Dashboard":

    st.title("👥 Employee Attrition Analytics Dashboard")

    st.write(
        "Analyze workforce characteristics and identify patterns "
        "associated with employee attrition."
    )

    st.divider()

    # -----------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------

    total_employees = len(df)
    employees_left = int(df["Attrition_Flag"].sum())
    employees_stayed = total_employees - employees_left
    attrition_rate = employees_left / total_employees * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "👥 Total Employees",
        total_employees
    )

    col2.metric(
        "❌ Employees Left",
        employees_left
    )

    col3.metric(
        "✅ Employees Stayed",
        employees_stayed
    )

    col4.metric(
        "📉 Attrition Rate",
        f"{attrition_rate:.2f}%"
    )

    st.divider()

    # -----------------------------------------------------
    # ATTRITION DISTRIBUTION
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        attrition_count = (
            df["Attrition"]
            .value_counts()
            .reset_index()
        )

        attrition_count.columns = [
            "Attrition",
            "Employees"
        ]

        fig = px.pie(
            attrition_count,
            names="Attrition",
            values="Employees",
            title="Employee Attrition Distribution",
            hole=0.4
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        department_data = (
            df.groupby("Department")["Attrition_Flag"]
            .mean()
            .reset_index()
        )

        department_data["Attrition Rate"] = (
            department_data["Attrition_Flag"] * 100
        )

        fig = px.bar(
            department_data,
            x="Department",
            y="Attrition Rate",
            title="Attrition Rate by Department",
            text_auto=".1f"
        )

        fig.update_yaxes(
            title="Attrition Rate (%)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -----------------------------------------------------
    # OVERTIME
    # -----------------------------------------------------

    overtime_data = (
        df.groupby("OverTime")["Attrition_Flag"]
        .mean()
        .reset_index()
    )

    overtime_data["Attrition Rate"] = (
        overtime_data["Attrition_Flag"] * 100
    )

    fig = px.bar(
        overtime_data,
        x="OverTime",
        y="Attrition Rate",
        title="Attrition Rate by Overtime",
        text_auto=".1f"
    )

    fig.update_yaxes(
        title="Attrition Rate (%)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# EMPLOYEE ANALYSIS
# =========================================================

elif page == "🔍 Employee Analysis":

    st.title("🔍 Employee Analysis")

    st.write(
        "Explore how employee characteristics are associated "
        "with attrition."
    )

    st.divider()

    # -----------------------------------------------------
    # FILTERS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        departments = st.multiselect(
            "Department",
            options=sorted(
                df["Department"].unique()
            ),
            default=sorted(
                df["Department"].unique()
            )
        )

    with col2:

        genders = st.multiselect(
            "Gender",
            options=sorted(
                df["Gender"].unique()
            ),
            default=sorted(
                df["Gender"].unique()
            )
        )

    with col3:

        overtime = st.multiselect(
            "Overtime",
            options=sorted(
                df["OverTime"].unique()
            ),
            default=sorted(
                df["OverTime"].unique()
            )
        )

    filtered_df = df[
        df["Department"].isin(departments)
        & df["Gender"].isin(genders)
        & df["OverTime"].isin(overtime)
    ]

    st.write(
        f"Showing **{len(filtered_df)} employees**"
    )

    st.divider()

    # -----------------------------------------------------
    # JOB SATISFACTION
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        satisfaction_data = (
            filtered_df.groupby("JobSatisfaction")
            ["Attrition_Flag"]
            .mean()
            .reset_index()
        )

        satisfaction_data["Attrition Rate"] = (
            satisfaction_data["Attrition_Flag"] * 100
        )

        fig = px.bar(
            satisfaction_data,
            x="JobSatisfaction",
            y="Attrition Rate",
            title="Job Satisfaction vs Attrition",
            text_auto=".1f"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -----------------------------------------------------
    # BUSINESS TRAVEL
    # -----------------------------------------------------

    with col2:

        travel_data = (
            filtered_df.groupby("BusinessTravel")
            ["Attrition_Flag"]
            .mean()
            .reset_index()
        )

        travel_data["Attrition Rate"] = (
            travel_data["Attrition_Flag"] * 100
        )

        fig = px.bar(
            travel_data,
            x="BusinessTravel",
            y="Attrition Rate",
            title="Business Travel vs Attrition",
            text_auto=".1f"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -----------------------------------------------------
    # AGE DISTRIBUTION
    # -----------------------------------------------------

    fig = px.box(
        filtered_df,
        x="Attrition",
        y="Age",
        color="Attrition",
        title="Age Distribution by Attrition"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -----------------------------------------------------
    # MONTHLY INCOME
    # -----------------------------------------------------

    fig = px.box(
        filtered_df,
        x="Attrition",
        y="MonthlyIncome",
        color="Attrition",
        title="Monthly Income by Attrition"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# ATTRITION PREDICTION
# =========================================================

elif page == "🤖 Attrition Prediction":

    st.title("🤖 Employee Attrition Prediction")

    st.write(
        "Enter employee information to estimate the probability "
        "of employee attrition using the trained XGBoost model."
    )

    st.divider()

    # Columns that were removed during model training
    columns_to_remove = [
        "EmployeeCount",
        "EmployeeNumber",
        "Over18",
        "StandardHours",
        "Attrition",
        "Attrition_Flag"
    ]

    feature_df = df.drop(
        columns=[
            col for col in columns_to_remove
            if col in df.columns
        ]
    )

    input_data = {}

    with st.form("prediction_form"):

    st.subheader("Employee Information")

    col1, col2 = st.columns(2)

    columns = feature_df.columns.tolist()

    for i, column in enumerate(columns):

        current_column = (
            col1 if i % 2 == 0 else col2
        )

        with current_column:

            # Check whether the column is numeric
            if pd.api.types.is_numeric_dtype(feature_df[column]):

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

            # Categorical feature
            else:

                options = sorted(
                    feature_df[column]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                input_data[column] = st.selectbox(
                    column,
                    options
                )

    predict_button = st.form_submit_button(
        "🔮 Predict Attrition"
    )

    # -----------------------------------------------------
    # PREDICTION RESULT
    # -----------------------------------------------------

    if predict_button:

        input_df = pd.DataFrame(
            [input_data]
        )

        prediction = model.predict(
            input_df
        )[0]

        probability = model.predict_proba(
            input_df
        )[0][1]

        risk = probability * 100

        st.divider()

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            if prediction == 1:

                st.error(
                    "⚠️ HIGH RISK — Employee is predicted "
                    "to leave."
                )

            else:

                st.success(
                    "✅ LOW RISK — Employee is predicted "
                    "to stay."
                )

        with col2:

            st.metric(
                "Attrition Probability",
                f"{risk:.2f}%"
            )

        st.progress(
            min(int(risk), 100)
        )

        # Risk category
        if risk < 30:

            risk_level = "Low Risk"

        elif risk < 60:

            risk_level = "Moderate Risk"

        elif risk < 80:

            risk_level = "High Risk"

        else:

            risk_level = "Very High Risk"

        st.info(
            f"Risk Classification: **{risk_level}**"
        )

        # -------------------------------------------------
        # BASIC RECOMMENDATIONS
        # -------------------------------------------------

        st.subheader("💡 HR Recommendations")

        recommendations = []

        if input_data.get("OverTime") == "Yes":

            recommendations.append(
                "Review workload and overtime requirements."
            )

        if input_data.get("JobSatisfaction", 5) <= 2:

            recommendations.append(
                "Consider an employee engagement "
                "or satisfaction assessment."
            )

        if input_data.get("EnvironmentSatisfaction", 5) <= 2:

            recommendations.append(
                "Evaluate the employee's workplace environment."
            )

        if input_data.get("YearsSinceLastPromotion", 0) >= 4:

            recommendations.append(
                "Review career progression and promotion opportunities."
            )

        if input_data.get("BusinessTravel") == "Travel_Frequently":

            recommendations.append(
                "Review travel workload and work-life balance."
            )

        if len(recommendations) == 0:

            recommendations.append(
                "No major risk indicators detected from "
                "the selected rule-based checks."
            )

        for recommendation in recommendations:

            st.write(
                "• " + recommendation
            )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "📈 Model Performance":

    st.title("📈 Machine Learning Model Performance")

    st.write(
        "Comparison of the machine-learning algorithms used "
        "for employee attrition prediction."
    )

    st.divider()

    # -----------------------------------------------------
    # MODEL COMPARISON
    # -----------------------------------------------------

    st.subheader("Model Comparison")

    st.info(
        "Enter the actual performance values obtained from "
        "your model.py output."
    )

    model_data = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost"
        ],

        "Accuracy": [
            0.7177,
            0.8401,
            0.8707
        ],

        "Precision": [
            0.32,
            0.50,
            0.80
        ],

        "Recall": [
            0.6809,
            0.3191,
            0.2553
        ],

        "F1 Score": [
            0.4354,
            0.3896,
            0.3871
        ],

        "ROC-AUC": [
            0.7823,
            0.7942,
            0.7845
        ]
    })

    st.dataframe(
        model_data,
        use_container_width=True
    )
    

    st.warning(
        "Replace the 0.00 values above with the actual "
        "metrics from your model comparison."
    )

    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    st.subheader(
        "🌟 Feature Importance"
    )

    st.write(
        """
        Feature importance indicates which variables were most
        influential in the XGBoost model's predictions. It does
        not imply that these variables causally produce attrition.
        """
    )

    try:

        fitted_preprocessor = (
            model.named_steps["preprocessor"]
        )

        classifier = (
            model.named_steps["classifier"]
        )

        feature_names = (
            fitted_preprocessor
            .get_feature_names_out()
        )

        importance = (
            classifier.feature_importances_
        )

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        })

        importance_df = (
            importance_df
            .sort_values(
                "Importance",
                ascending=False
            )
            .head(10)
        )

        fig = px.bar(
            importance_df.sort_values(
                "Importance"
            ),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 10 Features"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    except Exception:

        st.info(
            "Feature importance will appear if the loaded "
            "model contains the required preprocessing information."
        )


# =========================================================
# FOOTER
# =========================================================

st.sidebar.divider()

st.sidebar.caption(
    "Employee Attrition Analytics & Prediction System"
)

st.sidebar.caption(
    "Developed as a Data Science Mini Project"
)
