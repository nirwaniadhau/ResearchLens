from google import genai
from google.genai import types
import os


def create_gemini_client():

    gemini_api_key = os.getenv("GEMINI_API_KEY")

    return genai.Client(
        api_key=gemini_api_key
    )


def analyze_page_image(
    gemini_client,
    image_bytes
):
    """
    Analyze a rendered PDF page using Gemini vision.
    """

    prompt = """
You are analyzing a page from a research paper.

Describe the important visual information on this page that
may not be fully captured by OCR/text extraction.

Focus on:
- diagrams and their relationships
- charts and graphs
- tables and their important values
- figures and screenshots
- flowcharts or architectures
- captions and labels
- visual structure that helps understand the research

Do NOT simply repeat normal paragraph text.

Return a concise but informative description in Markdown.

If there is no meaningful visual information, say:
"No significant visual information."
"""

    response = gemini_client.models.generate_content(
        model="gemini-3.8-flash",
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/png"
            ),
            prompt
        ]
    )

    return response.text