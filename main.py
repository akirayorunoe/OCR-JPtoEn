from module.PDFTranslator import PdfTranslator

if __name__ == "__main__":
    input_pdf_path = "pdf\_106424_.pdf"
    output_pdf_path = "pdf\_106424_m.pdf"

    pdf_translator = PdfTranslator(input_pdf_path, output_pdf_path,target_language='en')
    pdf_translator.process_pdf()
