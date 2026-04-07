import fitz  # PyMuPDF

def extract_sanskrit_range(input_pdf, output_txt, start_page, end_page):
    """
    Extracts text from a specific range of pages.
    :param start_page: The first page to include (1-indexed)
    :param end_page: The last page to include (inclusive)
    """
    try:
        doc = fitz.open(input_pdf)
        total_pages = len(doc)
        
        # Validation to ensure the range is within the PDF's limits
        if start_page < 1 or end_page > total_pages:
            print(f"Error: Range {start_page}-{end_page} is out of bounds (1-{total_pages}).")
            return

        with open(output_txt, "w", encoding="utf-8") as f:
            # We subtract 1 from start_page for 0-indexing, 
            # and end_page is used as the upper bound for the range.
            for i in range(start_page - 1, end_page):
                page = doc.load_page(i)
                text = page.get_text()
                
                f.write(f"--- Page {i + 1} ---\n")
                f.write(text)
                f.write("\n\n")
        
        doc.close()
        print(f"Successfully extracted pages {start_page} to {end_page} to {output_txt}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

# Example: Extract only pages 5 through 15
extract_sanskrit_range("charaka_samhita.pdf", "charaka_samhitha.txt", 19, 20)