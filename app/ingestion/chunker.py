from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_documents(page_contents, pdf_path):

    documents = []

    for page in page_contents:

        combined_page_content = (
            f"--- PAGE {page['page_number']} ---\n\n"
            f"[TEXT]\n"
            f"{page['text']}\n"
        )

        if page["visual_description"]:

            combined_page_content += (
                "\n[VISUAL ANALYSIS]\n"
                f"{page['visual_description']}\n"
                "[END VISUAL ANALYSIS]\n"
            )

        document = Document(
            page_content=combined_page_content,
            metadata={
                "source": pdf_path,
                "page": page["page_number"]
            }
        )

        documents.append(document)

    return documents


def chunk_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=400
    )

    chunks = splitter.split_documents(
        documents
    )

    return chunks