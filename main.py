from module.PDFTranslator import PdfTranslator

if __name__ == "__main__":
    input_pdf_path = "/Users/innotech/Downloads/[106424].pdf"
    output_pdf_path = "/Users/innotech/Downloads/translated_pdfs/merged_translated.pdf"

    pdf_translator = PdfTranslator(input_pdf_path, output_pdf_path,target_language='en')
    pdf_translator.process_pdf()
