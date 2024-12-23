from module.PDFTranslator import PdfTranslator

if __name__ == "__main__":
    input_pdf_path = "D:/hoang/self-project/OCR-JPtoEn/[106424].pdf"
    output_pdf_path = "D:/hoang/self-project/OCR-JPtoEn/[106424]-translate.pdf"

    pdf_translator = PdfTranslator(input_pdf_path, output_pdf_path, target_language='en')
    pdf_translator.process_pdf()
