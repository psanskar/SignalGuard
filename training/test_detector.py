from detector.preprocessing import JobPreprocessor
from detector.predictor import JobPredictor
from detector.shap_explainer import SHAPExplainer


# -----------------------------------------
# Sample job
# -----------------------------------------

job = {
    "title": "Software Developer",

    "company_profile": """
    We are a growing technology company developing
    innovative software solutions.
    """,

    "description": """
    We are looking for a software developer to join
    our engineering team. You will work with experienced
    developers on real-world applications.
    """,

    "requirements": """
    Bachelor's degree in computer science or related field.
    Knowledge of Python, JavaScript and databases.
    """,

    "benefits": """
    Competitive salary, health insurance and paid leave.
    """,

    "location": "Mumbai, India",

    "department": "Engineering",

    "salary_range": "50000-80000",

    "employment_type": "Full-time",

    "required_experience": "1-3 years",

    "required_education": "Bachelor's Degree",

    "industry": "Technology",

    "function": "Engineering",

    "telecommuting": 0,

    "has_company_logo": 1,

    "has_questions": 1
}


# -----------------------------------------
# Initialize components
# -----------------------------------------

preprocessor = JobPreprocessor()

predictor = JobPredictor()

explainer = SHAPExplainer()


# -----------------------------------------
# Prepare job
# -----------------------------------------

X, combined_text, structured_data = (
    preprocessor.prepare_job(job)
)

print("\nFeature matrix shape:")
print(X.shape)


# -----------------------------------------
# Prediction
# -----------------------------------------

result = predictor.predict_with_threshold(
    X,
    threshold=0.50
)

print("\nPrediction:")
print(result)


# -----------------------------------------
# SHAP explanation
# -----------------------------------------

feature_names = list(
    preprocessor.vectorizer.get_feature_names_out()
) + list(
    preprocessor.structured_features
)

explanation = explainer.explain(
    X,
    feature_names,
    top_n=10
)

print("\nTop SHAP signals:")

for item in explanation[0]:

    print(
        f"{item['feature']:30s} "
        f"{item['contribution']:+.6f} "
        f"→ {item['direction'].upper()}"
    )