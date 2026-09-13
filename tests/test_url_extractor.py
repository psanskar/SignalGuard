from detector.url_extractor import (
    JobURLExtractor,
    URLExtractionError
)

extractor = JobURLExtractor()
url = "https://www.linkedin.com/jobs/view/4462457618/"

try:
    job_data = extractor.extract(url)

    print("\n" + "=" * 60)
    print("URL EXTRACTION SUCCESSFUL")
    print("=" * 60)

    for key, value in job_data.items():
        print(f"\n--- {key.upper()} ---")
        print(value)

    print("\n" + "=" * 60)

except URLExtractionError as error:
    print("\nURL extraction failed:")
    print(error)
except Exception as error:
    print("\nUnexpected error:")
    print(error)