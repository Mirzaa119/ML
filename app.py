from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


# ---------------------------------------------------------
# Application configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Outcome Predictor",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_outcome_model.joblib"
INFO_PATH = BASE_DIR / "app_information.joblib"


# ---------------------------------------------------------
# Load model and metadata
# ---------------------------------------------------------
@st.cache_resource
def load_application_files():
    model = joblib.load(MODEL_PATH)
    information = joblib.load(INFO_PATH)
    return model, information


try:
    model, app_information = load_application_files()
except FileNotFoundError:
    st.error(
        "The model or application information file could not be found. "
        "Place both joblib files in the same folder as app.py."
    )
    st.stop()
except Exception as error:
    st.error(f"Application files could not be loaded: {error}")
    st.stop()


categorical_features = app_information["categorical_features"]
numerical_features = app_information["numerical_features"]
categorical_options = app_information["categorical_options"]
numerical_settings = app_information["numerical_settings"]
model_classes = app_information["model_classes"]
final_metrics = app_information["final_metrics"]


# ---------------------------------------------------------
# Display helpers
# ---------------------------------------------------------
CATEGORY_LABELS = {
    "Marital status": {
        1: "Single", 2: "Married", 3: "Widower", 4: "Divorced",
        5: "Facto union", 6: "Legally separated"
    },
    "Application mode": {
        1: "General admission, first phase",
        2: "Ordinance No. 612/93",
        5: "Special admission, Azores",
        7: "Other higher-course holders",
        10: "Ordinance No. 854-B/99",
        15: "International student",
        16: "Special admission, Madeira",
        17: "General admission, second phase",
        18: "General admission, third phase",
        26: "Ordinance No. 533-A/99, Plan B2",
        27: "Ordinance No. 533-A/99, Plan B3",
        39: "Applicants aged over 23",
        42: "Transfer",
        43: "Change of course",
        44: "Technological specialisation diploma",
        51: "Change of institution or course",
        53: "Short-cycle diploma holder",
        57: "International change of institution or course"
    },
    "Course": {
        33: "Biofuel Production Technologies",
        171: "Animation and Multimedia Design",
        8014: "Social Service (evening attendance)",
        9003: "Agronomy", 9070: "Communication Design",
        9085: "Veterinary Nursing", 9119: "Informatics Engineering",
        9130: "Equinculture", 9147: "Management",
        9238: "Social Service", 9254: "Tourism", 9500: "Nursing",
        9556: "Oral Hygiene",
        9670: "Advertising and Marketing Management",
        9773: "Journalism and Communication", 9853: "Basic Education",
        9991: "Management (evening attendance)"
    },
    "Previous qualification": {
        1: "Secondary education", 2: "Bachelor's degree",
        3: "Higher education degree", 4: "Master's degree",
        5: "Doctorate", 6: "Attended higher education",
        9: "12th year not completed", 10: "11th year not completed",
        12: "Other 11th-year qualification", 14: "10th year",
        15: "10th year not completed",
        19: "Basic education, third cycle",
        38: "Basic education, second cycle",
        39: "Technological specialisation course",
        40: "Higher education, first-cycle degree",
        42: "Professional higher technical course",
        43: "Higher education, second-cycle master's"
    },
    "Nacionality": {
        1: "Portuguese", 2: "German", 6: "Spanish", 11: "Italian",
        13: "Dutch", 14: "English", 17: "Lithuanian", 21: "Angolan",
        22: "Cape Verdean", 24: "Guinean", 25: "Mozambican",
        26: "Santomean", 32: "Turkish", 41: "Brazilian",
        62: "Romanian", 100: "Moldovan", 101: "Mexican",
        103: "Ukrainian", 105: "Russian", 108: "Cuban",
        109: "Colombian"
    },
    "Daytime/evening attendance": {
        0: "Evening",
        1: "Daytime"
    },
    "Displaced": {
        0: "No",
        1: "Yes"
    },
    "Educational special needs": {
        0: "No",
        1: "Yes"
    },
    "Debtor": {
        0: "No",
        1: "Yes"
    },
    "Tuition fees up to date": {
        0: "No",
        1: "Yes"
    },
    "Gender": {
        0: "Female",
        1: "Male"
    },
    "Scholarship holder": {
        0: "No",
        1: "Yes"
    },
    "International": {
        0: "No",
        1: "Yes"
    },
    "Mother's occupation": {
        0: "Student", 1: "Management or senior official",
        2: "Professional or scientific occupation",
        3: "Technician or associate professional",
        4: "Administrative occupation", 5: "Service or sales worker",
        6: "Agriculture or fishery worker", 7: "Skilled trade worker",
        8: "Machine operator or assembler", 9: "Elementary occupation",
        10: "Armed forces", 90: "Other occupation",
        99: "Unknown occupation"
    },
    "Father's occupation": {
        0: "Student", 1: "Management or senior official",
        2: "Professional or scientific occupation",
        3: "Technician or associate professional",
        4: "Administrative occupation", 5: "Service or sales worker",
        6: "Agriculture or fishery worker", 7: "Skilled trade worker",
        8: "Machine operator or assembler", 9: "Elementary occupation",
        10: "Armed forces", 90: "Other occupation",
        99: "Unknown occupation"
    }
}

CATEGORY_LABELS["Mother's qualification"] = CATEGORY_LABELS[
    "Previous qualification"
]
CATEGORY_LABELS["Father's qualification"] = CATEGORY_LABELS[
    "Previous qualification"
]

BINARY_FEATURES = {
    "Daytime/evening attendance", "Displaced",
    "Educational special needs", "Debtor", "Tuition fees up to date",
    "Gender", "Scholarship holder", "International"
}


def category_label(feature, value):
    """Display friendly labels while preserving dataset codes."""
    value = value.item() if hasattr(value, "item") else value

    if feature == "Application order":
        return f"Preference {int(value) + 1}"

    mapping = CATEGORY_LABELS.get(feature, {})

    if value in mapping:
        return f"{mapping[value]} (code {value})"

    return f"Code {value}"


def default_category_index(feature):
    """Select a safe default option."""
    options = categorical_options[feature]

    if feature in BINARY_FEATURES and 0 in options:
        return options.index(0)

    return 0


def numerical_input(feature, key):
    """Create a numerical input using dataset-supported ranges."""
    settings = numerical_settings[feature]

    minimum = float(settings["minimum"])
    maximum = float(settings["maximum"])
    median = float(settings["median"])

    integer_features = {
        "Age at enrollment",
        "Curricular units 1st sem (credited)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (without evaluations)"
    }

    if feature in integer_features:
        return st.number_input(
            feature,
            min_value=int(minimum),
            max_value=int(maximum),
            value=int(round(median)),
            step=1,
            key=key
        )

    return st.number_input(
        feature,
        min_value=minimum,
        max_value=maximum,
        value=median,
        step=0.1,
        format="%.2f",
        key=key
    )


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 750;
            color: #60A5FA;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #CBD5E1;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .result-card {
            padding: 1.4rem;
            border-radius: 12px;
            background-color: #172554;
            color: #F8FAFC;
            border-left: 6px solid #60A5FA;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }

        .warning-card {
            padding: 1rem;
            border-radius: 10px;
            background-color: #422006;
            color: #FEF3C7;
            border-left: 5px solid #F59E0B;
        }

        div.stButton > button {
            width: 100%;
            background-color: #2563EB;
            color: white;
            font-weight: 650;
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------
st.sidebar.title("🎓 Navigation")

page = st.sidebar.radio(
    "Select a page",
    [
        "Home",
        "Predict Outcome",
        "Model Performance",
        "About and Limitations"
    ]
)

st.sidebar.divider()
st.sidebar.caption("COM763 Advanced Machine Learning")
st.sidebar.caption("Prediction point: End of first semester")


# ---------------------------------------------------------
# Home page
# ---------------------------------------------------------
if page == "Home":
    st.markdown(
        '<div class="main-title">Student Academic Outcome Predictor</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
            A machine-learning decision-support system for identifying
            students who may benefit from additional academic support.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "The model predicts one of three possible outcomes: "
        "Dropout, Enrolled or Graduate."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model", "Logistic Regression")

    with col2:
        st.metric(
            "Test Accuracy",
            f"{final_metrics['Accuracy']:.1%}"
        )

    with col3:
        st.metric(
            "Test Macro F1",
            f"{final_metrics['Macro F1']:.1%}"
        )

    st.subheader("How the system works")

    st.markdown(
        """
        1. Enter the student's enrolment and first-semester information.
        2. Review the values for accuracy.
        3. Select **Predict Academic Outcome**.
        4. Review the predicted outcome and class probabilities.
        5. Use the result only as one source of information for academic support.
        """
    )

    st.subheader("Prediction outcomes")

    outcome_col1, outcome_col2, outcome_col3 = st.columns(3)

    with outcome_col1:
        st.success("**Graduate**\n\nThe model predicts successful completion.")

    with outcome_col2:
        st.info("**Enrolled**\n\nThe student may remain actively enrolled.")

    with outcome_col3:
        st.warning("**Dropout risk**\n\nAdditional support may be beneficial.")

    st.markdown(
        """
        <div class="warning-card">
            <strong>Responsible-use notice:</strong>
            Predictions are estimates and may be incorrect. The system must
            not be used to automatically penalise, reject or exclude students.
            Qualified staff should review each case individually.
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# Prediction page
# ---------------------------------------------------------
if page == "Predict Outcome":
    st.title("Student Outcome Prediction")

    st.write(
        "Enter information available by the end of the student's first semester."
    )

    with st.form("prediction_form"):
        personal_tab, academic_tab, financial_tab, context_tab = st.tabs(
            [
                "Personal and Enrolment",
                "Academic Performance",
                "Financial and Support",
                "Economic Context"
            ]
        )

        user_input = {}

        with personal_tab:
            st.subheader("Personal and enrolment information")

            personal_features = [
                "Marital status",
                "Application mode",
                "Application order",
                "Course",
                "Daytime/evening attendance",
                "Previous qualification",
                "Nacionality",
                "Mother's qualification",
                "Father's qualification",
                "Mother's occupation",
                "Father's occupation",
                "Displaced",
                "Educational special needs",
                "Gender",
                "Age at enrollment",
                "International"
            ]

            left_column, right_column = st.columns(2)

            for index, feature in enumerate(personal_features):
                target_column = (
                    left_column if index % 2 == 0 else right_column
                )

                with target_column:
                    if feature in categorical_features:
                        options = categorical_options[feature]

                        user_input[feature] = st.selectbox(
                            feature,
                            options=options,
                            index=default_category_index(feature),
                            format_func=lambda value, current_feature=feature:
                                category_label(current_feature, value),
                            key=f"input_{feature}"
                        )
                    else:
                        user_input[feature] = numerical_input(
                            feature,
                            key=f"input_{feature}"
                        )

        with academic_tab:
            st.subheader("Admission and first-semester performance")

            academic_features = [
                "Previous qualification (grade)",
                "Admission grade",
                "Curricular units 1st sem (credited)",
                "Curricular units 1st sem (enrolled)",
                "Curricular units 1st sem (evaluations)",
                "Curricular units 1st sem (approved)",
                "Curricular units 1st sem (grade)",
                "Curricular units 1st sem (without evaluations)"
            ]

            left_column, right_column = st.columns(2)

            for index, feature in enumerate(academic_features):
                target_column = (
                    left_column if index % 2 == 0 else right_column
                )

                with target_column:
                    user_input[feature] = numerical_input(
                        feature,
                        key=f"input_{feature}"
                    )

            enrolled_units = user_input[
                "Curricular units 1st sem (enrolled)"
            ]

            approved_units = user_input[
                "Curricular units 1st sem (approved)"
            ]

            if enrolled_units > 0:
                approval_rate = approved_units / enrolled_units
            else:
                approval_rate = 0.0

            user_input["First semester approval rate"] = approval_rate

            st.metric(
                "Automatically calculated approval rate",
                f"{approval_rate:.1%}"
            )

        with financial_tab:
            st.subheader("Financial and support information")

            financial_features = [
                "Debtor",
                "Tuition fees up to date",
                "Scholarship holder"
            ]

            columns = st.columns(3)

            for index, feature in enumerate(financial_features):
                with columns[index]:
                    options = categorical_options[feature]

                    user_input[feature] = st.selectbox(
                        feature,
                        options=options,
                        index=default_category_index(feature),
                        format_func=lambda value, current_feature=feature:
                            category_label(current_feature, value),
                        key=f"input_{feature}"
                    )

        with context_tab:
            st.subheader("Economic context")

            st.caption(
                "These indicators describe the economic environment "
                "represented in the training dataset."
            )

            context_features = [
                "Unemployment rate",
                "Inflation rate",
                "GDP"
            ]

            columns = st.columns(3)

            for index, feature in enumerate(context_features):
                with columns[index]:
                    user_input[feature] = numerical_input(
                        feature,
                        key=f"input_{feature}"
                    )

        submitted = st.form_submit_button(
            "Predict Academic Outcome"
        )

    if submitted:
        validation_errors = []

        enrolled = user_input[
            "Curricular units 1st sem (enrolled)"
        ]

        evaluated = user_input[
            "Curricular units 1st sem (evaluations)"
        ]

        approved = user_input[
            "Curricular units 1st sem (approved)"
        ]

        if approved > enrolled:
            validation_errors.append(
                "Approved units cannot exceed enrolled units."
            )

        if approved > evaluated:
            validation_errors.append(
                "Approved units cannot exceed evaluated units."
            )

        if validation_errors:
            for validation_error in validation_errors:
                st.error(validation_error)
        else:
            required_columns = (
                categorical_features + numerical_features
            )

            input_dataframe = pd.DataFrame(
                [{
                    column: user_input[column]
                    for column in required_columns
                }]
            )

            try:
                prediction = model.predict(input_dataframe)[0]
                probabilities = model.predict_proba(input_dataframe)[0]

                probability_table = pd.DataFrame({
                    "Outcome": model.classes_,
                    "Probability": probabilities
                }).sort_values(
                    "Probability",
                    ascending=False
                )

                selected_probability = probability_table.loc[
                    probability_table["Outcome"] == prediction,
                    "Probability"
                ].iloc[0]

                st.markdown(
                    f"""
                    <div class="result-card">
                        <h3>Predicted outcome: {prediction}</h3>
                        <p>
                            Model probability:
                            <strong>{selected_probability:.1%}</strong>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if prediction == "Dropout":
                    st.warning(
                        "The student may benefit from additional academic, "
                        "financial or wellbeing support. A qualified member "
                        "of staff should review the case."
                    )
                elif prediction == "Enrolled":
                    st.info(
                        "The student is predicted to remain enrolled. "
                        "Continued monitoring and support may still be useful."
                    )
                else:
                    st.success(
                        "The model predicts that the student is likely "
                        "to graduate."
                    )

                chart_data = probability_table.set_index(
                    "Outcome"
                )[["Probability"]]

                st.subheader("Outcome probabilities")

                chart = (
                    alt.Chart(probability_table)
                    .mark_bar(color="#8ab4f8")
                    .encode(
                        x=alt.X(
                            "Outcome:N",
                            sort=None,
                            title="Outcome"
                        ),
                        y=alt.Y(
                            "Probability:Q",
                            scale=alt.Scale(domain=[0, 1]),
                            axis=alt.Axis(format="%"),
                            title="Probability"
                        ),
                        tooltip=[
                            alt.Tooltip(
                                "Outcome:N",
                                title="Outcome"
                            ),
                            alt.Tooltip(
                                "Probability:Q",
                                title="Probability",
                                format=".1%"
                            )
                        ]
                    )
                )

                st.altair_chart(chart, use_container_width=True)

                displayed_table = probability_table.copy()
                displayed_table["Probability"] = (
                    displayed_table["Probability"]
                    .map(lambda value: f"{value:.1%}")
                )

                st.dataframe(
                    displayed_table,
                    hide_index=True,
                    use_container_width=True
                )

                st.caption(
                    "Probabilities represent the model's relative confidence. "
                    "They do not guarantee a student's future outcome."
                )

            except Exception as error:
                st.error(f"Prediction could not be completed: {error}")

# ---------------------------------------------------------
# Performance page
# ---------------------------------------------------------
if page == "Model Performance":
    st.title("Model Performance")

    st.write(
        "The final tuned Logistic Regression model was evaluated on "
        "an untouched test set of 885 student records."
    )

    metric_columns = st.columns(5)

    displayed_metrics = [
        ("Accuracy", "Accuracy"),
        ("Macro Precision", "Macro Precision"),
        ("Macro Recall", "Macro Recall"),
        ("Macro F1", "Macro F1"),
        ("Weighted F1", "Weighted F1")
    ]

    for column, (label, metric_key) in zip(
        metric_columns,
        displayed_metrics
    ):
        with column:
            st.metric(
                label,
                f"{final_metrics[metric_key]:.1%}"
            )

    st.subheader("Class-level performance")

    class_results = pd.DataFrame({
        "Outcome": ["Dropout", "Enrolled", "Graduate"],
        "Precision": [0.8059, 0.3966, 0.8297],
        "Recall": [0.6725, 0.5912, 0.7715],
        "F1-score": [0.7332, 0.4747, 0.7995],
        "Test records": [284, 159, 442]
    })

    st.dataframe(
        class_results.style.format({
            "Precision": "{:.1%}",
            "Recall": "{:.1%}",
            "F1-score": "{:.1%}"
        }),
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Cross-validation and test performance")

    validation_comparison = pd.DataFrame({
        "Evaluation": [
            "Five-fold cross-validation",
            "Unseen test set"
        ],
        "Macro F1": [
            0.6793,
            final_metrics["Macro F1"]
        ]
    })

    st.bar_chart(
        validation_comparison.set_index("Evaluation")
    )

    st.info(
        "The Enrolled class has lower precision and F1-score than the "
        "Dropout and Graduate classes. Predictions for this class should "
        "therefore be interpreted with additional caution."
    )


# ---------------------------------------------------------
# About and limitations page
# ---------------------------------------------------------
if page == "About and Limitations":
    st.title("About and Limitations")

    st.subheader("System overview")

    st.markdown(
        """
        - **Task:** Multiclass student-outcome classification
        - **Model:** Tuned Logistic Regression
        - **Prediction point:** End of the first semester
        - **Outcomes:** Dropout, Enrolled and Graduate
        - **Dataset size:** 4,424 student records
        - **Final test size:** 885 records
        - **Second-semester variables:** Excluded
        """
    )

    st.subheader("Known limitations")

    st.markdown(
        """
        - The model was trained using data collected from one educational context.
        - Results may not generalise to other institutions, countries or periods.
        - The Enrolled class is more difficult for the model to identify accurately.
        - Incorrect predictions can occur for every outcome.
        - Social and economic circumstances may change after the recorded data.
        - Model probabilities should not be interpreted as certainty.
        - The system does not provide causal explanations.
        """
    )

    st.subheader("Appropriate use")

    st.markdown(
        """
        The system may be used as an additional indicator to help qualified
        academic-support staff identify students who may benefit from assistance.
        Predictions should be reviewed alongside current information and direct
        communication with the student.
        """
    )

    st.subheader("Inappropriate use")

    st.markdown(
        """
        The system should not be used to:

        - Automatically remove or exclude students
        - Make admission or disciplinary decisions
        - Deny financial or academic assistance
        - Replace professional academic judgement
        - Publicly label individual students
        - Guarantee whether a student will graduate or leave
        """
    )

    st.warning(
        "This application is an educational machine-learning demonstration, "
        "not an automated decision-making system."
    )