from detector.url_extractor import JobURLExtractor

extractor = JobURLExtractor()

url = "https://www.linkedin.com/jobs/view/4462457618/"

try:
    result = extractor.extract(url)

    print("\nURL extraction successful.\n")

    for key, value in result.items():
        print(f"{key}:")
        print(value)
        print()

except Exception as error:
    print("\nURL extraction failed.")
    print(error)