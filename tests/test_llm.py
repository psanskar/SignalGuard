from llm.llm_client import LLMClient


def main():

    llm = LLMClient()

    response = llm.generate(
        "Explain in one sentence what a fraudulent job posting is."
    )

    print("\nGemini Response:")
    print(response)


if __name__ == "__main__":
    main()