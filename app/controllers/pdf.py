import pdfplumber
from .preprocessing import remove_special_characters

def parse_pdf(path: str):
    with pdfplumber.open(path) as pdf:
        # first_page = pdf.pages[0]
        pages = {}
        for number, page in enumerate(pdf.pages):
            pages[number] = page.extract_text()
        return pages

# parse_pdf('/Users/SidharrthNagappan/Downloads/example_1.pdf')