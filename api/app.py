from flask import Flask, request, jsonify
from flask_cors import CORS

from detector.url_extractor import JobURLExtractor, URLExtractionError
from detector.risk_engine import RiskEngine


app = Flask(__name__)
CORS(app)

print("Starting SignalGuard API...")

risk_engine = RiskEngine()
url_extractor = JobURLExtractor()

print("SignalGuard API initialized successfully.")


REQUIRED_FIELDS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "name": "SignalGuard API",
        "status": "running",
        "version": "1.0"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy"
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:
        job_data = request.get_json()

        if not job_data:
            return jsonify({
                "error": "Request body must contain JSON data."
            }), 400

        # Check for missing required fields
        missing_fields = [
            field
            for field in REQUIRED_FIELDS
            if field not in job_data
        ]

        if missing_fields:
            return jsonify({
                "error": "Missing required fields.",
                "missing_fields": missing_fields
            }), 400

        # Check for empty required fields
        empty_fields = [
            field
            for field in REQUIRED_FIELDS
            if not str(job_data.get(field, "")).strip()
        ]

        if empty_fields:
            return jsonify({
                "error": "Required fields cannot be empty.",
                "empty_fields": empty_fields
            }), 400

        result = risk_engine.assess(job_data)

        return jsonify(result), 200

    except Exception as error:

        print(f"Prediction error: {error}")

        return jsonify({
            "error": "Unable to assess the job posting.",
            "details": str(error)
        }), 500


@app.route("/analyze-url", methods=["POST"])
def analyze_url():

    try:
        data = request.get_json()

        if not data or not data.get("url"):
            return jsonify({
                "error": "URL is required."
            }), 400

        url = data["url"]

        # Extract the job posting
        job_data = url_extractor.extract(url)

        # Run the normal SignalGuard analysis
        result = risk_engine.assess(job_data)

        # Preserve the source URL
        result["source_url"] = job_data.get("source_url", url)

        # Return the original extracted posting content
        # so the frontend can render the Job Posting preview.
        result["extracted_preview"] = {
            "title": job_data.get("title", ""),
            "company_profile": job_data.get("company_profile", ""),
            "description": job_data.get("description", ""),
            "requirements": job_data.get("requirements", ""),
            "benefits": job_data.get("benefits", "")
        }

        return jsonify(result), 200

    except URLExtractionError as error:

        print(f"URL extraction error: {error}")

        return jsonify({
            "error": str(error)
        }), 400

    except Exception as error:

        print(f"URL analysis error: {error}")

        return jsonify({
            "error": "Unable to analyze the job URL.",
            "details": str(error)
        }), 500


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )