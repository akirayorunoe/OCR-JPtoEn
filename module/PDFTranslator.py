import os
import fitz  # PyMuPDF

from module.ImageTextExtractor import ImageTextExtractor
from module.LanguageTranslator import LanguageTranslator

class PdfTranslator:
    def __init__(self, input_pdf_path, output_pdf_path,target_language='en'):
        self.input_pdf_path = input_pdf_path
        self.output_pdf_path = output_pdf_path
        self.image_text_extractor = ImageTextExtractor()
        self.translator = LanguageTranslator()
        self.target_language = target_language

    def process_pdf(self):
        # Open the input PDF document
        pdf_document = fitz.open(self.input_pdf_path)

        # Create the output directory if it doesn't exist
        output_directory = os.path.dirname(self.output_pdf_path)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Create a new PDF document for translated text
        pdf_with_translated_text = fitz.open()

        for page_number in range(53,54):  # pdf_document.page_count
            # Extract images from the current page of the PDF
            page = pdf_document[page_number]
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                # Extract image data
                image_index = img[0]
                base_image = pdf_document.extract_image(image_index)
                image_bytes = base_image["image"]

                # Save the image to a temporary file
                image_path = f"temp_image_{page_number}_{img_index}.jpg"
                with open(image_path, "wb") as image_file:
                    image_file.write(image_bytes)

                # Extract text from the image and translate
                text = self.image_text_extractor.extract_text_from_image(image_path)
                translated_text = self.translator.translate_text(text,self.target_language)

                # Add the translated text to the new PDF document
                pdf_with_translated_text.new_page(width=int(page.rect.width) * 2, height=int(page.rect.height))
                page_translated = pdf_with_translated_text[-1]

                # Insert the original image on the left side of the PDF page
                image_ref = page.get_images(full=True)[img_index][0]
                base_image = pdf_document.extract_image(image_ref)
                image_bytes = base_image["image"]
                pix_original = fitz.Pixmap(image_bytes)
                rect_original = fitz.Rect(0, 0, page.rect.width, page.rect.height)
                page_translated.insert_image(rect_original, pixmap=pix_original)

                # Insert the original text on the right side of the PDF page
                rect_original_text = fitz.Rect(page.rect.width, 0, page.rect.width * 1.5, page.rect.height)
                font = fitz.Font("cjk")
                page.insert_font(fontname="japan", fontbuffer=font.buffer)
                page_translated.insert_text((int(page.rect.width) + 10, 10), 'Original text', fontname='tiro', fontsize=10)

                original_text_lines = text.split('\n')
                max_line_width = int(page.rect.width * 0.5)  # Adjust the maximum line width as needed

                for i, line in enumerate(original_text_lines):
                    font_size = 6
                    while fitz.get_text_length(line, fontname='japan', fontsize=font_size) > max_line_width:
                        font_size -= 1

                    page_translated.insert_text(
                        (int(page.rect.width) + 10, 20 + i * 8),  # Adjust the vertical position as needed
                        line,
                        fontname='japan',
                        fontsize=font_size
                    )

                # Insert the translated text on the right side of the PDF page
                rect_translated = fitz.Rect(page.rect.width * 1.5, 0, page.rect.width * 2, page.rect.height)
                page_translated.insert_text((int(page.rect.width) * 1.5 + 10, 10), 'Translated text', fontname='tiro', fontsize=10)

                translated_text_lines = translated_text.split('\n')
                
                for i, line in enumerate(translated_text_lines):
                    font_size = 6
                    while fitz.get_text_length(line, fontname='tiro', fontsize=font_size) > max_line_width:
                        font_size -= 1

                    page_translated.insert_text(
                        (int(page.rect.width) * 1.5 + 10, 20 + i * 8),  # Adjust the vertical position as needed
                        line,
                        fontname='tiro',
                        fontsize=font_size
                    )
                # Remove temporary image file
                os.remove(image_path)
    
        # Save the new PDF document (use garbage=4, deflate=True to down size the file)
        pdf_with_translated_text.save(self.output_pdf_path, garbage=4, deflate=True)

        # Close the PDF documents
        pdf_document.close()
        pdf_with_translated_text.close()
