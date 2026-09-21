# SignalGuard

**AI-powered fraudulent job posting risk assessment using machine learning, explainability, safety knowledge retrieval, and LLM-generated guidance.**

SignalGuard is a web-based system that analyzes job postings for potential fraud and scam indicators. It combines a machine-learning classifier with rule-based red-flag detection, SHAP explainability, safety-focused knowledge retrieval, and Gemini-generated explanations.

The machine-learning model is the **primary fraud classifier**. Other components provide additional evidence, context, explanations, and safety guidance rather than replacing the classifier.

---

## Features

* **Job URL analysis** — analyze supported public job-posting URLs.
* **Manual job analysis** — paste job details directly into SignalGuard.
* **Machine-learning classification** — TF-IDF + structured features with XGBoost.
* **Red-flag detection** — identifies common job-scam patterns such as upfront payments, financial-information requests, fake checks, cryptocurrency requests, urgency, and unrealistic income claims.
* **Negation handling** — reduces false positives for legitimate statements such as "we do not charge fees."
* **SHAP explanations** — shows which model features influenced the ML prediction.
* **Safety-focused RAG** — retrieves supporting job-safety knowledge relevant to detected concerns.
* **Gemini explanations** — generates a human-readable explanation and practical safety recommendations.
* **Risk engine** — combines ML probability and red-flag severity into an overall risk level.
* **Responsive frontend** — designed for desktop and mobile use.

---

## System Architecture

```text
Job URL / Manual Job Details
            ↓
Preprocessing + Feature Extraction
            ↓
      TF-IDF + Structured Features
            ↓
        XGBoost Classifier
            ↓
      ┌─────┴─────┐
      ↓           ↓
 Red-Flag       SHAP
 Detector    Explanation
      │           │
      └─────┬─────┘
            ↓
     RAG Safety Retrieval
            ↓
        Risk Engine
            ↓
      Gemini Explanation
            ↓
       Frontend Results
```

The components have different responsibilities:

| Component         | Purpose                                                                 |
| ----------------- | ----------------------------------------------------------------------- |
| XGBoost           | Primary fraudulent-job classifier                                       |
| Red-flag detector | Detects explicit scam indicators                                        |
| SHAP              | Explains model influence                                                |
| RAG               | Provides supporting safety knowledge                                    |
| Risk engine       | Produces overall LOW / MEDIUM / HIGH assessment                         |
| Gemini            | Converts evidence into a human-readable explanation and recommendations |

---

## Detection Pipeline

### 1. Input

SignalGuard accepts either:

* A supported job-posting URL
* Manually entered job information

For URL analysis, the system attempts to extract available job fields such as title, company, description, location, employment information, and other publicly available content.

Unavailable fields are left empty rather than being invented.

### 2. Preprocessing

Job information is cleaned and transformed into model-ready features.

Text is combined into a unified representation and processed using TF-IDF.

Structured features are also extracted to provide additional information to the classifier.

### 3. Machine-Learning Classification

SignalGuard uses an **XGBoost classifier** trained using:

* TF-IDF text features
* Unigram and bigram representations
* Structured job-posting features

The final model uses a decision threshold of **0.55**.

The ML probability represents the model's estimated likelihood of the fraudulent class. It should not be interpreted as proof that a job posting is fraudulent.

### 4. Red-Flag Detection

SignalGuard independently checks for explicit warning signs.

Current categories include:

* Upfront payment
* Financial information requests
* Sensitive personal information
* Cryptocurrency requests
* Fake checks
* Messaging-platform recruitment
* Urgency or pressure
* Unrealistic income claims

The detector includes negation handling to distinguish suspicious statements from legitimate warnings.

For example:

```text
"We do not charge any application fees."
```

should not be treated the same as:

```text
"Pay a registration fee before your application can be processed."
```

### 5. SHAP Explainability

SHAP is used to explain the influence of model features on the ML prediction.

A positive SHAP contribution pushes the model toward the fraudulent class.

A negative SHAP contribution pushes the model toward the legitimate class.

SHAP represents **model influence, not proof of fraud**.

### 6. Safety Knowledge Retrieval

SignalGuard uses a semantic retrieval layer based on:

```text
all-MiniLM-L6-v2
```

The RAG knowledge base contains safety information covering topics such as:

* Financial information
* Upfront payments
* Fake checks
* Cryptocurrency scams
* Messaging-platform scams
* Urgency pressure
* Unrealistic income
* General job safety

RAG output is supporting safety knowledge, **not another fraud probability**.

### 7. Risk Engine

The final risk level is determined from both ML output and red-flag severity.

```text
HIGH
  fraud_probability >= 0.75
  OR any HIGH red flag
  OR at least 2 MEDIUM red flags

MEDIUM
  fraud_probability >= 0.50
  OR at least 1 MEDIUM red flag

LOW
  otherwise
```

This intentionally allows a job with a relatively low ML probability to receive a **HIGH overall risk** when a serious explicit warning sign is detected.

### 8. Gemini Explanation

Gemini receives the available analysis evidence and produces:

* Summary
* Risk explanation
* Key concerns
* Safety recommendations
* Confidence note

A fallback explanation is used when the LLM service is unavailable.

---

## Model Performance

The final model was evaluated using a separate train / validation / test workflow.

| Accuracy  | 98.57% |
| Precision | 90.67% |
| Recall    | 78.61% |
| F1-score  | 84.21% |
| ROC-AUC   | 98.63% |
| PR-AUC    | 91.71% |

The dataset contains:

```text
Total postings:    17,880
Legitimate:        17,014
Fraudulent:           866
Fraud proportion:    ~4.84%
```

Because the dataset is imbalanced, precision, recall, F1-score, ROC-AUC, and PR-AUC are reported alongside accuracy.

---

## API

SignalGuard exposes a Flask API.

### `GET /`

Basic API/root response.

### `GET /health`

Health-check endpoint.

### `POST /predict`

Analyzes manually supplied job information.

### `POST /analyze-url`

Extracts available information from a job URL and analyzes the resulting posting.

Example development server:

```text
http://127.0.0.1:5000
```

---

## Project Structure

```text
SignalGuard/
│
├── api/
│   └── app.py
│
├── data/
│   └── processed/
│
├── detector/
│   ├── predictor.py
│   ├── preprocessing.py
│   ├── red_flags.py
│   ├── risk_engine.py
│   ├── shap_explainer.py
│   └── url_extractor.py
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── llm/
│   ├── explainer.py
│   ├── llm_client.py
│   └── prompt_builder.py
│
├── models/
│   ├── structured_features.joblib
│   ├── tfidf_vectorizer_structured.joblib
│   ├── threshold.joblib
│   └── xgb_structured_model.joblib
│
├── rag/
│   ├── knowledge/
│   ├── knowledge_base.py
│   ├── rag_engine.py
│   ├── retriever.py
│   └── semantic_retriever.py
│
├── tests/
│
├── training/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/psanskar/SignalGuard.git
cd SignalGuard
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a local `.env` file containing the required Gemini API configuration.

Do **not** commit `.env` to GitHub.

---

## Running SignalGuard

### Start the backend

From the project root:

```powershell
python -m api.app
```

The API runs at:

```text
http://127.0.0.1:5000
```

### Start the frontend

In a second terminal:

```powershell
python -m http.server 5500 --directory frontend
```

Open:

```text
http://127.0.0.1:5500/
```

---

## Testing

The repository contains tests covering:

* API behavior
* URL extraction
* Red-flag detection
* Negation handling
* Risk-engine behavior
* SHAP explanations
* RAG retrieval
* RAG edge cases
* Knowledge-base behavior
* LLM explanation components
* Prompt construction

Training and evaluation utilities are available under:

```text
training/
```

---

## Safety and Interpretation

SignalGuard is intended as a **job-safety assessment tool**, not as an authoritative fraud determination.

A HIGH risk result means that the available evidence contains strong indicators that warrant caution. A LOW result does not guarantee that a posting is legitimate.

Users should independently verify:

* The employer
* The company's official website and contact details
* The job through trusted channels
* Requests for money or sensitive information
* Payment instructions and financial requests

SignalGuard should support human decision-making rather than replace it.

---

## Limitations

* URL extraction depends on the information publicly exposed by the target website.
* Some job platforms may restrict or limit publicly available posting content.
* ML predictions depend on the training data and may produce false positives or false negatives.
* Red-flag rules identify known patterns and cannot cover every possible scam.
* SHAP explains model behavior but does not establish factual truth.
* RAG provides supporting safety information rather than independent verification.
* Gemini-generated explanations are dependent on the available evidence and model availability.

---

## Future Improvements

Potential future work includes:

* Broader job-platform URL support
* Improved extraction of dynamically rendered job pages
* Additional training data
* Continuous model evaluation
* More sophisticated calibration and threshold analysis
* Expanded safety knowledge coverage
* Deployment as a production web service
* Additional employer and domain verification signals

---

## Disclaimer

SignalGuard is an academic and experimental project designed to assist with fraudulent job-posting risk assessment. It does not guarantee that a job posting is legitimate or fraudulent, and users should perform independent verification before sharing sensitive information, making payments, or accepting employment.
