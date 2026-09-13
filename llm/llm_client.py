import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


class LLMUnavailableError(Exception):
    """Raised when all configured LLM models are unavailable."""
    pass


class LLMClient:

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Make sure it is set in your .env file."
            )

        self.client = genai.Client(api_key=api_key)

        # Try the preferred model first, then fallbacks.
        self.models = [
            "gemini-3.6-flash",
            "gemini-2.5-flash",
            "gemini-flash-lite-latest"
        ]

        self.max_retries = 2

        print("Gemini LLM client initialized successfully.")
        print(f"LLM models: {self.models}")

    def generate_json(self, prompt):

        last_error = None

        for model in self.models:

            for attempt in range(1, self.max_retries + 1):

                try:

                    print(
                        f"Calling Gemini model: {model} "
                        f"(attempt {attempt}/{self.max_retries})"
                    )

                    response = self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json"
                        )
                    )

                    if not response.text:
                        raise ValueError(
                            "Gemini returned an empty response."
                        )

                    result = json.loads(response.text)

                    if not isinstance(result, dict):
                        raise ValueError(
                            "Gemini response is not a JSON object."
                        )

                    print(f"Gemini response received from {model}.")

                    return result

                except Exception as error:

                    last_error = error

                    status_code = (
                        getattr(error, "status_code", None)
                        or getattr(error, "code", None)
                    )

                    print(
                        f"Gemini error using {model}: "
                        f"{type(error).__name__}: {error}"
                    )

                    # Retry temporary server/rate-limit errors.
                    if status_code in {429, 500, 502, 503, 504}:

                        if attempt < self.max_retries:
                            wait_time = 2 ** (attempt - 1)

                            print(
                                f"Temporary Gemini error. "
                                f"Retrying in {wait_time} second(s)..."
                            )

                            time.sleep(wait_time)

                        continue

                    # Invalid JSON or other non-transient error.
                    # Move to the next model.
                    break

        raise LLMUnavailableError(
            "All configured Gemini models failed. "
            f"Last error: {last_error}"
        )