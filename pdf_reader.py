import os
from dotenv import load_dotenv
from llama_parse import LlamaParse

load_dotenv()

api_key = os.getenv("LLAMA_CLOUD_API_KEY")

parser = LlamaParse(
    api_key=api_key,
    result_type="markdown"
)

documents = parser.load_data("D:\\ResearchLens\\data\\DLP Detecting Sensitive Information Leakage.docx (1).pdf")

print(type(documents))
print("Number of documents:", len(documents))

for document in documents:
    print("Type:", type(document))
    print("Metadata:", document.metadata)
    print("Text length:", len(document.text))
    print("-" * 50)