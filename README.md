# Student Academic Outcome Predictor

A Streamlit machine-learning application that predicts a student's academic outcome using enrolment, demographic, financial and first-semester academic information.

The system predicts one of three outcomes:

- Dropout
- Enrolled
- Graduate

## Project Objective

The application demonstrates how machine learning can support the early identification of students who may benefit from additional academic, financial or wellbeing assistance.

The prediction is made at the end of the first semester. All second-semester academic variables were excluded to prevent the system from using information that would not be available at the selected prediction point.

## Dataset

The project uses the **Predict Students' Dropout and Academic Success** dataset from the UCI Machine Learning Repository.

- Records: 4,424 students
- Original input variables: 36
- Target variable: `Target`
- Target classes: Dropout, Enrolled and Graduate
- Missing values: 0
- Duplicate records: 0

Dataset source:

[UCI Predict Students' Dropout and Academic Success Dataset](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

### Target Distribution

| Outcome | Records | Percentage |
|---|---:|---:|
| Graduate | 2,209 | 49.93% |
| Dropout | 1,421 | 32.12% |
| Enrolled | 794 | 17.95% |

## Prediction Point

The model predicts academic outcomes using information available by the end of the student's first semester.

The following second-semester variables were excluded:

- Curricular units 2nd semester credited
- Curricular units 2nd semester enrolled
- Curricular units 2nd semester evaluations
- Curricular units 2nd semester approved
- Curricular units 2nd semester grade
- Curricular units 2nd semester without evaluations

## Machine-Learning Workflow

The project includes the following stages:

1. Dataset loading and inspection
2. Missing-value and duplicate checks
3. Target distribution analysis
4. Exploratory data analysis
5. Feature selection
6. First-semester approval-rate feature engineering
7. Stratified training and testing split
8. Numerical and categorical preprocessing
9. Five-fold stratified cross-validation
10. Model comparison
11. Hyperparameter tuning
12. Unseen test-set evaluation
13. Permutation feature importance
14. Model persistence using Joblib
15. Streamlit application development

## Data Preprocessing

Numerical variables are processed using:

- Median imputation
- Standard scaling

Categorical variables are processed using:

- Most-frequent-value imputation
- One-hot encoding
- Unknown-category handling

The preprocessing operations and classifier are stored together in a Scikit-learn pipeline.

## Models Evaluated

The following models were compared:

- Dummy classifier
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Macro F1 was used as the primary model-selection metric because the target classes are imbalanced and performance across all three classes is important.

## Final Model

The deployed model is a tuned Logistic Regression classifier with:

- Regularisation parameter `C = 0.1`
- Balanced class weights
- LBFGS solver
- Maximum 3,000 iterations

The tuned Random Forest produced a higher cross-validation Macro F1, but it also showed a substantially larger difference between training and validation performance. Logistic Regression was selected for deployment because it demonstrated a smaller generalisation gap and provides a more interpretable baseline for an educational decision-support application.

## Final Test Performance

The final model was evaluated using an untouched test set containing 885 student records.

| Metric | Score |
|---|---:|
| Accuracy | 70.7% |
| Macro Precision | 67.7% |
| Macro Recall | 67.8% |
| Macro F1 | 66.9% |
| Weighted F1 | 72.0% |

### Class-Level Performance

| Outcome | Precision | Recall | F1-score | Test records |
|---|---:|---:|---:|---:|
| Dropout | 80.6% | 67.3% | 73.3% | 284 |
| Enrolled | 39.7% | 59.1% | 47.5% | 159 |
| Graduate | 83.0% | 77.2% | 80.0% | 442 |

The Enrolled class has lower precision and F1-score than the Dropout and Graduate classes. Enrolled predictions should therefore be interpreted with additional caution.

## Important Predictive Features

Permutation importance identified the following influential features:

- First-semester approved curricular units
- Tuition fees up to date
- First-semester approval rate
- First-semester evaluations
- First-semester enrolled units
- Course
- First-semester grade
- Debtor status
- Scholarship status
- Age at enrolment

Feature importance indicates predictive contribution and does not demonstrate causation.

## Try the deployed application here:

[Open the Student Academic Outcome Predictor](https://student-outcome-predictorr.streamlit.app)


## Streamlit Application

The application contains four pages:

### Home

Provides an overview of the prediction system, model and responsible-use requirements.

### Predict Outcome

Collects student information through four input sections:

- Personal and enrolment information
- Academic performance
- Financial and support information
- Economic context

The application returns:

- Predicted academic outcome
- Probability for each outcome
- Outcome probability chart
- Support-oriented interpretation message

### Model Performance

Displays final evaluation metrics, class-level results and cross-validation performance.

### About and Limitations

Explains the prediction point, appropriate use, inappropriate use and known limitations.

## Project Structure

```text
student-outcome-predictor/
├── app.py
├── app_information.joblib
├── data.csv
├── requirements.txt
├── student_outcome_analysis.ipynb
├── student_outcome_model.joblib
├── README.md
└── .gitignore
```

### File Descriptions

| File | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `app_information.joblib` | Application metadata, feature settings and evaluation metrics |
| `data.csv` | Original student dataset |
| `requirements.txt` | Required Python packages |
| `student_outcome_analysis.ipynb` | Complete machine-learning workflow |
| `student_outcome_model.joblib` | Trained preprocessing and classification pipeline |
| `.gitignore` | Files excluded from Git version control |

## Installation

Python 3.10 or later is recommended.

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/student-outcome-predictor.git
cd student-outcome-predictor
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in the default web browser.

## Requirements

The main dependencies are:

- Streamlit
- Pandas
- NumPy
- Altair
- Scikit-learn 1.6.1
- Joblib

## Known Limitations

- The dataset represents a specific educational context.
- Performance may not generalise to other institutions, countries or time periods.
- The Enrolled class is more difficult for the model to classify accurately.
- Incorrect predictions can occur for every outcome.
- Student circumstances may change after the recorded information was collected.
- Model probabilities represent relative confidence rather than certainty.
- The model identifies statistical patterns and does not establish causal relationships.
- Some demographic and socioeconomic variables may introduce fairness concerns.

## Responsible Use

This application is an educational machine-learning demonstration and decision-support tool.

Predictions should:

- Be reviewed by qualified academic-support staff
- Be considered alongside current student information
- Be used to offer supportive interventions
- Include direct communication with the student

Predictions should not be used to:

- Automatically exclude or penalise students
- Make admission or disciplinary decisions
- Deny academic or financial assistance
- Replace professional judgement
- Publicly label individual students
- Guarantee whether a student will graduate or leave

## Author

**Name:** Mirza
**Student ID:**  
**Module:** COM763 Advanced Machine Learning