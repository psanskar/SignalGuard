import re
import html
import joblib
import pandas as pd
from scipy.sparse import hstack


class JobPreprocessor:

    def __init__(
        self,
        vectorizer_path="models/tfidf_vectorizer_structured.joblib",
        structured_features_path="models/structured_features.joblib"
    ):
        """
        Load the same TF-IDF vectorizer and structured feature
        configuration that were used during model training.
        """

        self.vectorizer = joblib.load(vectorizer_path)
        self.structured_features = joblib.load(
            structured_features_path
        )

    # -----------------------------------------
    # Text cleaning
    # -----------------------------------------

    def clean_text(self, text):

        if pd.isna(text):
            return ""

        text = str(text)

        # Decode HTML entities such as &amp;
        text = html.unescape(text)

        # Replace URL placeholders
        text = re.sub(
            r"#URL_\d+#",
            " URL ",
            text
        )

        # Replace actual URLs
        text = re.sub(
            r"https?://\S+|www\.\S+",
            " URL ",
            text
        )

        # Remove HTML tags
        text = re.sub(
            r"<[^>]+>",
            " ",
            text
        )

        # Normalize whitespace
        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        return text

    # -----------------------------------------
    # Prepare a job
    # -----------------------------------------

    def prepare_job(self, job):

        """
        Convert a job dictionary into model-ready features.

        Expected input example:

        {
            "title": "Software Developer",
            "company_profile": "...",
            "description": "...",
            "requirements": "...",
            "benefits": "...",
            "location": "...",
            "department": "...",
            "salary_range": "...",
            "employment_type": "...",
            "required_experience": "...",
            "required_education": "...",
            "industry": "...",
            "function": "...",
            "telecommuting": 0,
            "has_company_logo": 1,
            "has_questions": 0
        }
        """

        text_columns = [
            "title",
            "company_profile",
            "description",
            "requirements",
            "benefits"
        ]

        cleaned = {}

        # Clean text fields
        for column in text_columns:

            cleaned[column] = self.clean_text(
                job.get(column, "")
            )

        # -----------------------------------------
        # Combined text
        # -----------------------------------------

        combined_text = " ".join(
            cleaned[column]
            for column in text_columns
        )

        # -----------------------------------------
        # Text statistics
        # -----------------------------------------

        text_length = len(combined_text)

        word_count = len(
            combined_text.split()
        )

        url_count = len(
            re.findall(
                r"https?://\S+|www\.\S+",
                " ".join(
                    str(job.get(column, ""))
                    for column in text_columns
                )
            )
        )

        # -----------------------------------------
        # Missing text fields
        # -----------------------------------------

        missing_text_fields = sum(
            1
            for column in [
                "company_profile",
                "description",
                "requirements",
                "benefits"
            ]
            if not cleaned[column]
        )

        # -----------------------------------------
        # Structured features
        # -----------------------------------------

        structured_data = {
            "text_length": text_length,
            "word_count": word_count,
            "url_count": url_count,
            "missing_text_fields": missing_text_fields,
            "telecommuting": self.to_binary(
                job.get("telecommuting", 0)
            ),
            "has_company_logo": self.to_binary(
                job.get("has_company_logo", 0)
            ),
            "has_questions": self.to_binary(
                job.get("has_questions", 0)
            )
        }

        structured_values = pd.DataFrame(
            [structured_data]
        )[self.structured_features].astype(float).values

        # -----------------------------------------
        # TF-IDF
        # -----------------------------------------

        X_text = self.vectorizer.transform(
            [combined_text]
        )

        # -----------------------------------------
        # Combine features
        # -----------------------------------------

        X = hstack([
            X_text,
            structured_values
        ]).tocsr()

        return X, combined_text, structured_data

    # -----------------------------------------
    # Convert values to binary
    # -----------------------------------------

    @staticmethod
    def to_binary(value):

        if value in [1, True, "1", "true", "True"]:
            return 1

        return 0