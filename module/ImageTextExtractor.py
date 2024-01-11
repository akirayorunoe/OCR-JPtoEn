# # # Using easyocr
# # class ImageTextExtractor:
# #     def __init__(self):
# #         # Create an EasyOCR reader
# #         self.reader = easyocr.Reader(['ja'], gpu=False)  # You can adjust languages and GPU usage

# #     def preprocess_image(self, image_path):
# #         # Read the image
# #         img = cv2.imread(image_path)
# #         return img

# #     def extract_text_from_image(self, image_path):
# #         # Preprocessing using OpenCV
# #         img = self.preprocess_image(image_path)

# #         # Use EasyOCR for OCR
# #         result = self.reader.readtext(img)
        
# #         # Extract and concatenate text from the result with newline characters
# #         text_lines = [text_info[1] for text_info in result]
# #         text = ''.join(text_lines)

# #         return text.strip()


import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pytesseract
from PIL import ImageFont, ImageDraw, Image
from pytesseract import Output

class ImageTextExtractor:
    def __init__(self):
        os.environ['TESSDATA_PREFIX'] = 'tessdata'
        pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
        self.custom_config = r'--oem 3 --psm 4 -l jpn_best+osd -c chop_enable=T -c use_new_state_cost=F -c segment_segcost_rating=F -c enable_new_segsearch=0 -c language_model_ngram_on=0 -c textord_force_make_prop_words=F -c edges_max_children_per_outline=50'

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        invert = np.invert(img)

        # Chuyển ảnh sang đen trắng
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Áp dụng GaussianBlur để giảm nhiễu
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Invert the image
        inverted = cv2.bitwise_not(invert)
        
        # Extract the saturation channel (assuming the image is in BGR format)
        _, saturation, _ = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Áp dụng adaptive threshold để tạo ảnh nhị phân
        # thresh = cv2.adaptiveThreshold(invert, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        img_resize = cv2.resize(img,(0,0),fx=1.25,fy=1.25)

        # Sử dụng morphology để loại bỏ nhiễu và đặc biệt là kết cấu văn bản
        # Taking a matrix of size 5 as the kernel 
        kernel = np.ones((5, 5), np.uint8)  
        # morph = cv2.morphologyEx(img_resize, cv2.MORPH_CLOSE, kernel, iterations=2)
        # img_erosion = cv2.erode(img_resize, kernel, iterations=1) 
        # img_dilation = cv2.dilate(img_resize, kernel, iterations=1) 
        
        # sharpening
        kernel_S = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharp = cv2.filter2D(img_resize, -1, kernel_S)

        # cv2.imshow("sharp", sharp)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.waitKey(1)
        
        return img_resize
    
    def find_paragraph(self, image):
        inverted = cv2.bitwise_not(image)
        gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
        # create background image
        bg = cv2.dilate(gray, np.ones((5,5), dtype=np.uint8))
        bg = cv2.GaussianBlur(bg, (5,5), 1)
        # subtract out background from source
        src_no_bg = 255-cv2.absdiff(gray, bg)

        # Load image, grayscale, Gaussian blur, Otsu's threshold
        
        blur = cv2.GaussianBlur(src_no_bg, (5,5), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


        # Create rectangular structuring element and dilate
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        dilate = cv2.dilate(thresh, kernel, iterations=4)

        # Find contours and draw rectangle
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
            roi = image[y:y + h, x:x + w]  # Extract the region of interest
            text = pytesseract.image_to_string(roi, config = self.custom_config)
            # print(f"Bounding Box: ({x}, {y}, {w}, {h}), Text: {text}")
            self.extracted_texts.append(text)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        # cv2.imshow('thresh', thresh)
        # cv2.imshow('src_no_bg', src_no_bg)
        # cv2.imshow('dilate', dilate)
        # cv2.imshow('image', image)
        # cv2.waitKey()
        return image

    def extract_text_from_image(self, image_path):
        img = self.preprocess_image(image_path)
        # Các tham số được thêm vào cấu hình (https://github.com/tesseract-ocr/tessdoc/blob/main/tess3/ControlParams.md)
        
        # Initialize the list to store extracted texts
        self.extracted_texts = []

        self.rect_img(img)
        combined_text = ''.join(self.extracted_texts[::-1])
 
        # text = pytesseract.image_to_string(img, config = self.custom_config)

        # print(combined_text)
        return combined_text.strip()

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
        # for z, a in enumerate(data.splitlines()):
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

    pdf_translator = ImageTextExtractor()
    pdf_translator.extract_text_from_image('/Users/innotech/Desktop/OCR-JPtoEn/temp_image_0_0.jpg')

###       Note:
#         # The --oem parameter specifies the OCR Engine Mode, which determines which OCR engine Tesseract should use. There are several OEM modes available:

#         # 0: Original Tesseract only.
#         # 1: Neural nets LSTM only.
#         # 2: Legacy OCR engine only.
#         # 3: Both LSTM and legacy OCR engines. (Default)
#         # For example, setting --oem 1 indicates the use of the neural nets LSTM OCR engine.

#         # PSM (Page Segmentation Mode):

#         # The --psm parameter specifies the Page Segmentation Mode, which defines how Tesseract should interpret the layout of the image. It determines how the OCR engine should treat the input image in terms of text layout.

#         # 0: Orientation and script detection (OSD) only.
#         # 1: Automatic page segmentation with OSD.
#         # 2: Automatic page segmentation, but no OSD or OCR.
#         # 3: Fully automatic page segmentation, but no OSD. (Default)
#         # 4: Assume a single column of text of variable sizes.
#         # 5: Assume a single uniform block of vertically aligned text.
#         # 6: Assume a single uniform block of text.
#         # 7: Treat the image as a single text line.
#         # 8: Treat the image as a single word.
#         # 9: Treat the image as a single word in a circle.
#         # 10: Treat the image as a single character.
#         # 11: Sparse text. Find as much text as possible in no particular order.
#         # 12: Sparse text with OSD.
#         # 13: Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.
#         # Use Tesseract for OCR
