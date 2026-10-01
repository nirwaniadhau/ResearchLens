import os
from llama_cloud import LlamaCloud


def parse_pdf(pdf_path):
    """
    Parse a PDF using LlamaCloud and return
    the parsed pages.
    """

    llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

    client = LlamaCloud(
        api_key=llama_cloud_api_key
    )

    print("\n========================================")
    print("STARTING PDF PARSING")
    print("========================================")

    file = client.files.create(
        file=pdf_path,
        purpose="parse"
    )

    print("✓ PDF uploaded")
    print("File ID:", file.id)

    result = client.parsing.parse(
        file_id=file.id,
        tier="agentic",
        version="latest",
        expand=["markdown"]
    )

    print("✓ PDF parsed successfully")

    parsed_pages = result.markdown.pages

    print(
        "Number of parsed pages:",
        len(parsed_pages)
    )

    return parsed_pages