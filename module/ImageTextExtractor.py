import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pytesseract
from PIL import ImageFont, ImageDraw, Image
from pytesseract import Output
import easyocr

class ImageTextExtractor:
    def __init__(self):
        os.environ['TESSDATA_PREFIX'] = 'tessdata'
        pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\tessdata'
        self.custom_config = r'--oem 3 --psm 4 -l jpn_fast+osd -c chop_enable=T -c use_new_state_cost=F -c segment_segcost_rating=F -c enable_new_segsearch=0 -c language_model_ngram_on=0 -c textord_force_make_prop_words=F -c edges_max_children_per_outline=50'
        # self.reader = easyocr.Reader(['ja', 'en'])

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
        
        return img_resize
    
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
        
    def find_paragraph(self, image):
        inverted = cv2.bitwise_not(image)
        gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
        # create background image
        bg = cv2.dilate(gray, np.ones((6,6), dtype=np.uint8))
        # bg = cv2.GaussianBlur(bg, (5,5), 1)
        # subtract out background from source
        src_no_bg = 255-cv2.absdiff(gray, bg)
        src_no_bg = self.thick_font(src_no_bg)
        src_no_bg = self.thin_font(src_no_bg)
        # sharpening
        kernel_S = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        src_no_bg = cv2.filter2D(src_no_bg, -1, kernel_S)
        src_no_bg = cv2.filter2D(src_no_bg, -1, kernel_S)
        # src_no_bg = self.noise_removal(src_no_bg)

        # Load image, grayscale, Gaussian blur, Otsu's threshold
        
        blur = cv2.GaussianBlur(src_no_bg, (3,3), 0)
        thresh = cv2.threshold(src_no_bg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


        # Create rectangular structuring element and dilate
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        dilate = cv2.dilate(morph, kernel, iterations=9)

        # Để tách hai đoạn văn bản cạnh nhau thành hai đoạn riêng biệt
        # Giảm kích thước kernel khi tạo structuring element cho dilate và morphologyEx
        # Tăng giá trị của iterations cho morphologyEx và dilate
        # Find contours and draw rectangle
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            # print(cv2.contourArea(c))
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
            roi = image[y:y + h, x:x + w]  # Extract the region of interest
            if not self.is_bright(roi):
                # roi = cv2.bitwise_not(roi)
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(roi, (1,1), 0)
                blur = cv2.medianBlur(blur,1)
                th3 = cv2.threshold(blur,180,255,cv2.THRESH_BINARY)[1]
                # wide = cv2.Canny(blurred, 10, 200)
                # tight = cv2.Canny(th3, 225, 255)
                kernel_S = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                roi = cv2.filter2D(th3, -1, kernel_S)
               
                # roi = self.auto_canny(roi)
                
                # show the images
                # cv2.imshow("roi", roi)
                # cv2.imshow("Edges", np.hstack([wide, tight, auto]))
                # cv2.waitKey()
            text = pytesseract.image_to_string(roi, config = self.custom_config)
            # print(f"Bounding Box: ({x}, {y}, {w}, {h}), Text: {text}")
            self.extracted_texts.append(text)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        # cv2.imshow('thresh', thresh)
        # cv2.imshow('src_no_bg', src_no_bg)
        # cv2.imshow('morph', morph)
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
        return combined_text

    # def extract_text_from_image_easyocr(self, image_path):
    #     img = self.preprocess_image(image_path)
    #     # Các tham số được thêm vào cấu hình (https://github.com/tesseract-ocr/tessdoc/blob/main/tess3/ControlParams.md)
        
    #     # Initialize the list to store extracted texts
    #     self.extracted_texts = []

    #     img = self.find_paragraph(img)
    #     result = self.reader.readtext(img,paragraph=True, x_ths=0.5)
    #     text_lines = [text_info[1] for text_info in result]
    #     text = ''.join(text_lines)
    #     # text = pytesseract.image_to_string(img, config = self.custom_config)

    #     print(text)
    #     return text.strip()
    
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
    pdf_translator.extract_text_from_image('/Users/innotech/Desktop/OCR-JPtoEn/temp_image_53_0.jpg')

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
