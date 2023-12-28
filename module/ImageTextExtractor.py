# import os
# import cv2
# import numpy as np
# import pytesseract
# import matplotlib.pyplot as plt
# from PIL import Image
# import easyocr

# #Using pytesseract
# class ImageTextExtractor:
#     def __init__(self):
#         # Set TESSDATA_PREFIX
#         os.environ['TESSDATA_PREFIX'] = 'tessdata'
#         pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

#     def preprocess_image(self, image_path):
#         # Đọc ảnh
#         img = cv2.imread(image_path)

#         return img

#     def extract_text_from_image(self, image_path):
#         # Preprocessing using OpenCV
#         img = self.preprocess_image(image_path)

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
#         text = pytesseract.image_to_string(img, lang='jpn+jpn_vert+jpn_vert_new1+jpn_vert_new2+jpn_ver5+eng+equ', config='--oem 3 --psm 4')
#         return text.strip()

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
from PIL import Image
from pytesseract import Output

class ImageTextExtractor:
    def __init__(self):
        os.environ['TESSDATA_PREFIX'] = 'tessdata'
        pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

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
        img_resize = cv2.resize(blurred,(0,0),fx=1,fy=1)

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
        
        return img
        
    def extract_text_from_image(self, image_path):
        img = self.preprocess_image(image_path)
        
        # Các tham số được thêm vào cấu hình (https://github.com/tesseract-ocr/tessdoc/blob/main/tess3/ControlParams.md)
        custom_config = r'--oem 3 --psm 4 -l jpn_best+osd -c chop_enable=T -c use_new_state_cost=F -c segment_segcost_rating=F -c enable_new_segsearch=0 -c language_model_ngram_on=0 -c textord_force_make_prop_words=F -c edges_max_children_per_outline=50'
        # self.rect_img(img)
        text = pytesseract.image_to_string(img,config=custom_config)

        print(text)
        return text.strip()

    def rect_img(self, img):
        d = pytesseract.image_to_data(img, output_type=Output.DICT, config=r'--psm 6')
        n_boxes = len(d['level'])
        
        for i in range(n_boxes):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            
            # Lọc các bounding box có kích thước phù hợp
            if w > 50 and h > 20:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

# custom_config = r'--oem 3 --psm 4 -l jpn_fast+jpn_vert_fast+osd  -c tessedit_create_boxfile=1 -c chop_enable=T -c use_new_state_cost=F -c segment_segcost_rating=F -c enable_new_segsearch=0 -c language_model_ngram_on=0 -c textord_force_make_prop_words=F -c edges_max_children_per_outline=100'

