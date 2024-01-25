import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pytesseract
from PIL import ImageFont, ImageDraw, Image
from pytesseract import Output
import easyocr

class ImageTextExtractorEasyOCR:
    def __init__(self):
        os.environ['TESSDATA_PREFIX'] = 'tessdata'
        pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
        self.custom_config = r'--oem 3 --psm 4 -l jpn_fast+osd -c chop_enable=T -c use_new_state_cost=F -c segment_segcost_rating=F -c enable_new_segsearch=0 -c language_model_ngram_on=0 -c textord_force_make_prop_words=F -c edges_max_children_per_outline=50'
        self.reader = easyocr.Reader(['ja', 'en'])

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
        img_resize = cv2.resize(img,(0,0),fx=1.25,fy=1.25,interpolation = cv2.INTER_CUBIC)

        # Sử dụng morphology để loại bỏ nhiễu và đặc biệt là kết cấu văn bản
        # Taking a matrix of size 5 as the kernel 
        kernel = np.ones((5, 5), np.uint8)  
        # morph = cv2.morphologyEx(img_resize, cv2.MORPH_CLOSE, kernel, iterations=2)
        # img_erosion = cv2.erode(img_resize, kernel, iterations=1) 
        # img_dilation = cv2.dilate(img_resize, kernel, iterations=1) 
        
        # sharpening
        # kernel_S = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        kernel_S = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharp = cv2.filter2D(img_resize, -1, kernel_S)

        # cv2.imshow("sharp", sharp)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.waitKey(1)
        
        return img
    
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
        
    def extract_text_from_image(self, image_path):
        img = self.preprocess_image(image_path)
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

    pdf_translator = ImageTextExtractorEasyOCR()
    pdf_translator.extract_text_from_image('/Users/innotech/Desktop/OCR-JPtoEn/temp_image_53_0.jpg')

