import os
import fitz  # PyMuPDF
from io import BytesIO
from typing import List, Tuple, Dict

from module.ImageTextExtractor import ImageTextExtractor
from module.ImageTextExtractorEasyOCR import ImageTextExtractorEasyOCR
from module.LanguageTranslator import LanguageTranslator

class PdfTranslator:
    def __init__(self, input_pdf_path: str, output_pdf_path: str, target_language: str = 'en'):
        self.input_pdf_path = input_pdf_path
        self.output_pdf_path = output_pdf_path
        self.image_text_extractor = ImageTextExtractorEasyOCR()
        self.translator = LanguageTranslator()
        self.target_language = target_language
        self._setup_output_directory()

    def _setup_output_directory(self) -> None:
        output_directory = os.path.dirname(self.output_pdf_path)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    def process_pdf(self):
        with fitz.open(self.input_pdf_path) as pdf_document, \
             fitz.open() as pdf_with_translated_text:
            
            for page_number in range(pdf_document.page_count):  # 5,6
                page = pdf_document[page_number]
                image_list = page.get_images(full=True)

                for img_index, img in enumerate(image_list):
                    # Extract and process image using BytesIO
                    image_index = img[0]
                    base_image = pdf_document.extract_image(image_index)
                    image_bytes = base_image["image"]

                    # Process image directly from memory
                    with BytesIO(image_bytes) as image_stream:
                        text_instances, (img_width, img_height) = self.image_text_extractor.extract_text_with_position(image_stream)

                        # Create new page with double width
                        pdf_with_translated_text.new_page(
                            width=int(page.rect.width) * 2,
                            height=int(page.rect.height)
                        )
                        page_translated = pdf_with_translated_text[-1]

                        # Insert original images efficiently
                        pix_original = fitz.Pixmap(image_bytes)
                        try:
                            # Original image on left
                            rect_original = fitz.Rect(0, 0, page.rect.width, page.rect.height)
                            page_translated.insert_image(rect_original, pixmap=pix_original)
                            
                            # Translated image on right
                            rect_translated = fitz.Rect(page.rect.width, 0, page.rect.width * 2, page.rect.height)
                            page_translated.insert_image(rect_translated, pixmap=pix_original)
                        finally:
                            pix_original = None  # Help garbage collection

                        # Calculate scaling factors
                        scale_x = page.rect.width / img_width
                        scale_y = page.rect.height / img_height

                        # Process text instances
                        for text_instance in text_instances:
                            self._process_text_instance(text_instance, page_translated, page.rect.width, scale_x, scale_y)

            # Save with optimization
            pdf_with_translated_text.save(
                self.output_pdf_path,
                garbage=4,
                deflate=True,
                clean=True
            )

    def _process_text_instance(self, text_instance: Dict, page_translated: fitz.Page, 
                             page_width: float, scale_x: float, scale_y: float) -> None:
        original_text = text_instance['text']
        translated_text = self.translator.translate_text(original_text, self.target_language)
        
        # Calculate scaled coordinates
        bbox = text_instance['bbox']
        scaled_bbox = [(p[0] * scale_x, p[1] * scale_y) for p in bbox]
        top_left, _, bottom_right, _ = scaled_bbox
        
        width = text_instance['width'] * scale_x
        height = text_instance['height'] * scale_y
        angle = text_instance.get('angle', 0)
        
        # Create rectangle for text overlay
        padding = 1
        rect = fitz.Rect(
            top_left[0] + page_width - padding,
            top_left[1] - padding,
            bottom_right[0] + page_width + padding,
            bottom_right[1] + padding
        )
        
        # Draw white background
        page_translated.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))
        
        # Calculate optimal font size
        font_size = height * 0.8
        text_width = fitz.get_text_length(translated_text, fontname='tiro', fontsize=font_size)
        
        while text_width > width and font_size > 4:
            font_size *= 0.9
            text_width = fitz.get_text_length(translated_text, fontname='tiro', fontsize=font_size)
        
        # Insert text with proper orientation
        if angle != 0:  # Vertical text
            center_x = (top_left[0] + bottom_right[0]) / 2 + page_width
            center_y = (top_left[1] + bottom_right[1]) / 2
            page_translated.draw_text(
                (center_x, center_y),
                translated_text,
                fontname='tiro',
                fontsize=font_size,
                rotate=90,
                render_mode=0
            )
        else:  # Horizontal text
            x = top_left[0] + page_width + (width - text_width) / 2
            y = top_left[1] + height * 0.7
            page_translated.insert_text(
                (x, y),
                translated_text,
                fontname='tiro',
                fontsize=font_size,
                render_mode=0
            )
