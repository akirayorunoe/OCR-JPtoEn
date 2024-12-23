import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pytesseract
from PIL import ImageFont, ImageDraw, Image
from pytesseract import Output
import easyocr
import torch
import math
from io import BytesIO

class ImageTextExtractorEasyOCR:
    def __init__(self):
        os.environ['TESSDATA_PREFIX'] = 'tessdata'
        pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
        self.custom_config = r'--oem 3 --psm 4 -l jpn_fast+osd -c chop_enable=T -c use_new_state_cost=F -c segment_segcost_rating=F -c enable_new_segsearch=0 -c language_model_ngram_on=0 -c textord_force_make_prop_words=F -c edges_max_children_per_outline=50'
        # Initialize EasyOCR with better detection parameters
        self.reader = easyocr.Reader(['ja', 'en'], gpu=True if torch.cuda.is_available() else False, 
                                   model_storage_directory='./models',
                                   download_enabled=True,
                                   recog_network='japanese_g2')

    def _read_image(self, image_input):
        """Helper method to read image from either file path or bytes"""
        if isinstance(image_input, str):
            # If input is a file path
            return cv2.imread(image_input)
        elif isinstance(image_input, BytesIO):
            # If input is BytesIO object
            image_input.seek(0)
            file_bytes = np.asarray(bytearray(image_input.read()), dtype=np.uint8)
            return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        elif isinstance(image_input, bytes):
            # If input is raw bytes
            file_bytes = np.asarray(bytearray(image_input), dtype=np.uint8)
            return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            raise ValueError("Unsupported image input type")

    def preprocess_image(self, image_input):
        # Read image using the helper method
        img = self._read_image(image_input)
        if img is None:
            raise ValueError("Could not read image")
            
        # Store original image
        original = img.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Check if image is bright or dark and adjust preprocessing accordingly
        is_bright = self.is_bright(gray)
        
        # Create different versions of the image for better text detection
        preprocessed_versions = []
        
        # Version 1: Enhanced contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(gray)
        preprocessed_versions.append(contrast_enhanced)
        
        # Version 2: Binarization for dark text on light background
        if is_bright:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        preprocessed_versions.append(binary)
        
        # Version 3: Adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
        preprocessed_versions.append(adaptive_thresh)
        
        # Version 4: Color-based preprocessing for colored text
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:,:,1]
        v_channel = hsv[:,:,2]
        preprocessed_versions.append(s_channel)
        preprocessed_versions.append(v_channel)
        
        return {
            'original': original,
            'versions': preprocessed_versions,
            'is_bright': is_bright
        }

    def extract_text_with_position(self, image_input):
        """
        Enhanced text extraction with better handling of different fonts, sizes, and colors
        """
        # Preprocess image with multiple versions
        processed_images = self.preprocess_image(image_input)
        
        # Get image dimensions
        img_height, img_width = processed_images['original'].shape[:2]
        
        # Configure reader parameters for better detection
        reader_kwargs = {
            'decoder': 'beamsearch',
            'beamWidth': 5,
            'batch_size': 8,
            'contrast_ths': 0.1,  # Lower threshold for better detection of low-contrast text
            'adjust_contrast': 0.5,
            'width_ths': 0.5,  # More tolerant width threshold
            'height_ths': 0.5   # More tolerant height threshold
        }
        
        all_results = []
        # Process each version of the preprocessed image
        for idx, img_version in enumerate(processed_images['versions']):
            try:
                results = self.reader.readtext(img_version, **reader_kwargs)
                all_results.extend(results)
            except Exception as e:
                print(f"Error processing version {idx}: {str(e)}")
                continue
        
        # Merge and deduplicate results
        text_instances = []
        seen_boxes = set()
        
        for (bbox, text, prob) in all_results:
            # Create a hashable version of the bbox for deduplication
            bbox_key = tuple(tuple(point) for point in bbox)
            
            if bbox_key in seen_boxes or prob < 0.3:  # Skip duplicates and very low confidence
                continue
                
            seen_boxes.add(bbox_key)
            
            # Get coordinates and calculate dimensions
            top_left, top_right, bottom_right, bottom_left = bbox
            w = abs(top_right[0] - top_left[0])
            h = abs(bottom_left[1] - top_left[1])
            
            # Determine text orientation
            # More sophisticated angle detection
            dx = top_right[0] - top_left[0]
            dy = top_right[1] - top_left[1]
            angle = math.degrees(math.atan2(dy, dx))
            if abs(angle) > 45:  # Vertical text
                angle = 90
            else:  # Horizontal text
                angle = 0
            
            # Calculate relative size for better font size estimation
            relative_size = (w * h) / (img_width * img_height)
            
            text_instances.append({
                'text': text,
                'bbox': bbox,
                'width': int(w),
                'height': int(h),
                'angle': angle,
                'confidence': prob,
                'relative_size': relative_size,
                'estimated_font_size': int(h * 0.7)  # Rough estimate of font size based on height
            })
        
        # Sort by confidence
        text_instances.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Return both text instances and image dimensions
        return text_instances, (img_width, img_height)

    def thin_font(self,image):
        image = cv2.bitwise_not(image)
        kernel = np.ones((2,2),np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.bitwise_not(image)
        return (image)
    
    def thick_font(self,image):
        image = cv2.bitwise_not(image)
        kernel = np.ones((2,2),np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        image = cv2.bitwise_not(image)
        return (image)
    
    def noise_removal(self, image):
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.medianBlur(image, 3)
        return (image)
    
    def is_bright(self,image):
        blur = cv2.blur(image, (5, 5))  # With kernel size depending upon image size
        if cv2.mean(blur)[0] > 127:  # The range for a pixel's value in grayscale is (0-255), 127 lies midway
            return True # (127 - 255) denotes light image
        else:
            return False
        
    def auto_canny(self, image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
        # return the edged image
        return edged
        
    def extract_text_from_image(self, image_input):
        img = self.preprocess_image(image_input)
        # Các tham số được thêm vào cấu hình (https://github.com/tesseract-ocr/tessdoc/blob/main/tess3/ControlParams.md)
        
        # Initialize the list to store extracted texts
        self.extracted_texts = []

        self.rect_img(img)

        combined_text = ''.join(self.extracted_texts)
 
        # text = pytesseract.image_to_string(img, config = self.custom_config)

        # print(combined_text)
        return combined_text

    def find_paragraph(self, image):
        result = self.reader.readtext(image, paragraph=True,x_ths=0.5)
    
        # Lặp qua các vùng chứa văn bản
        for detection in result:
            # Lấy tọa độ của các điểm
            points = detection[0]
            
            x_min, y_min = points[0]
            x_max, y_max = points[2]
            
            w, h = x_max - x_min, y_max - y_min
            roi = image[y_min:y_min + h, x_min:x_min + w]
            
            if not self.is_bright(roi):
                roi = cv2.resize(roi,(0,0),fx=2,fy=2,interpolation = cv2.INTER_CUBIC)
                # roi = cv2.bitwise_not(roi)
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(roi, (1,1), 0)
                blur = cv2.medianBlur(blur,1)
                th3 = cv2.threshold(blur,180,255,cv2.THRESH_BINARY)[1]
                # wide = cv2.Canny(blurred, 10, 200)
                # tight = cv2.Canny(th3, 225, 255)
                kernel_S = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                roi = cv2.filter2D(th3, -1, kernel_S)
            else:
                roi = cv2.resize(roi,(0,0),fx=1.25,fy=1.25,interpolation = cv2.INTER_CUBIC)
            text = pytesseract.image_to_string(roi, config = self.custom_config)
            # print(f"Bounding Box: ({x}, {y}, {w}, {h}), Text: {text}")
            self.extracted_texts.append(text)
            # cv2.imshow('roi', roi)
            # cv2.waitKey()
        # Vẽ bounding box và hiển thị
        for detection in result:
            # detection[0]: Bounding box coordinates
            # detection[1]: Extracted text
            box = detection[0]
            text = detection[1]

            # Lấy các đỉnh của bounding box
            (top_left, top_right, bottom_right, bottom_left) = box

            # Chuyển đổi tọa độ thành số nguyên
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))

            # Vẽ bounding box lên hình ảnh
            image = cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

            # # Hiển thị văn bản cùng với bounding box
            # cv2.putText(img, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        # cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        # cv2.imshow('thresh', thresh)
        # cv2.imshow('src_no_bg', src_no_bg)
        # cv2.imshow('morph', morph)
        # cv2.imshow('dilate', dilate)
        # cv2.imshow('image', image)
        # cv2.waitKey()
        return image
    
    def rect_img(self, img):
        box = pytesseract.image_to_boxes(img, config = " -c tessedit_create_boxfile=1")
        data = pytesseract.image_to_data(img, config = " -c tessedit_create_boxfile=1")
        hImg, wImg, _ = img.shape
        ### For character
        # for a in box.splitlines():
        #     a = a.split()
        #     x, y = int(a[1]), int(a[2])
        #     w, h = int(a[3]), int(a[4])

        #     cv2.rectangle(img, (x, hImg - y), (w, hImg - h), (255,0,0), 1)
        ###

        ### For word
        # for z, a in enumerate(data.splitlines()))):
        #     if z != 0:
        #         a = a.split()
        #         if len(a) == 12:
        #             x, y = int(a[6]), int(a[7])
        #             w, h = int(a[8]), int(a[9])
        #             cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 1)
            
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        ###

        ### For paragraph
        self.find_paragraph(img)
        ###

# custom_config = r'--oem 3 --psm 4 -l jpn_fast+jpn_vert_fast+osd  -c tessedit_create_boxfile=1 -c chop_enable=T -c use_new_state_cost=F -c segment_segcost_rating=F -c enable_new_segsearch=0 -c language_model_ngram_on=0 -c textord_force_make_prop_words=F -c edges_max_children_per_outline=100'

if __name__ == "__main__":
    input_pdf_path = "/Users/innotech/Downloads/[106424].pdf"
    output_pdf_path = "/Users/innotech/Downloads/translated_pdfs/merged_translated2.pdf"

    pdf_translator = ImageTextExtractorEasyOCR()
    pdf_translator.extract_text_from_image('/Users/innotech/Desktop/OCR-JPtoEn/temp_image_53_0.jpg')
