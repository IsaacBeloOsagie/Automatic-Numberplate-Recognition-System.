import base64
import io
from PIL import Image
from pytesseract import pytesseract
import cv2
import numpy as np
import imutils
from db_config import config
from myconnection import connect_to_mysql

class AnprsService:
    def __init__(self, file_image) :
        file = file_image.read()
        npimg = np.frombuffer(file, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)  

        # 1b.  Get the image type (e.g: jpg, jpeg, png, gif)
        image_type = file_image.mimetype.split("/")[1] #image/png -> split value = {0: image, 1: png}

        self.image = image
        self.image_type = image_type.upper()
        self.file_image = file_image;

    def image_to_base64(self, image) :
        image = Image.fromarray(image.astype("uint8"))
        rawBytes = io.BytesIO()
        image.save(rawBytes, self.image_type)
        rawBytes.seek(0)
        base64_image = base64.b64encode(rawBytes.read())
        return str(base64_image)

    def process_image(self) :
        """# 1. Grayscale and Blur"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # print("step 1")
        # print(self.image_to_base64(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)))

        """**2. Apply filter and find edges for localization**"""
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
        edged = cv2.Canny(bfilter, 30, 200) #Edge detection

        # print("step 2")     
        # print(self.image_to_base64(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB)))

        """## 3. Find Contours and Apply Mask"""
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break

        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0,255, -1)
        new_image = cv2.bitwise_and(self.image, self.image, mask=mask)

        # print("step 3") 
        # print(self.image_to_base64(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)))

        (x,y) = np.where(mask==255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2+1, y1:y2+1]
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

        """## 4. Use Tesseract To Read Text From Cropped Image"""
        path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        pytesseract.tesseract_cmd = path_to_tesseract

        text = pytesseract.image_to_string(cropped_image)

        if text == "" :
            print("Fall back 1 : to try and extrtact from the new image!")
            text = pytesseract.image_to_string(new_image)
        if text == "" :
            print("Fall back 2 :  Using Original Image To Try Extract Text!")
            text = pytesseract.image_to_string(self.image)

        response = {
            "file_name" : self.file_image.filename,
            "mime_type" : self.file_image.mimetype,
            "image_base64" : self.image_to_base64(self.image),
            "cropped_image_base64" : self.image_to_base64(cropped_image),
            "extracted_text": text
        }

        # Insert the processing record into DB before returning the result to Controller and subsequently to REACT.
        record_id = self.insert_into_DB(response)

        response["id"] = record_id
        return response


    def insert_into_DB(self, record) :
        insert_statement = ("INSERT INTO processed_image "
                             "(file_name, mime_type, image_base64, cropped_image_base64, extracted_text) " 
                             "VALUES (%(file_name)s, %(mime_type)s, %(image_base64)s, %(cropped_image_base64)s, %(extracted_text)s)");

        db_cnx = connect_to_mysql(config)
        if db_cnx and db_cnx.is_connected() :
            cursor = db_cnx.cursor()

            # Insert new record
            cursor.execute(insert_statement, record)
            insert_id = cursor.lastrowid

            # Make sure data is committed to the database
            db_cnx.commit()

            cursor.close()
            db_cnx.close()
            return insert_id
        else :
            print("Could not connect")
        return 0
