from PIL import Image
import pytesseract
import cv2
import os
import re


def ocr():
    try:
        image =  cv2.imread('/home/akhtech94/pythonprojects/yolov4/darknet/numberplate.jpg')        
        image = cv2.resize(image, None, fx=4.25, fy=4.25, interpolation=cv2.INTER_AREA)
        cv2.imshow("image", image)        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        res, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))        
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for x in contours:
            print(cv2.boundingRect(x)[1])

        contours = sorted(contours, key=lambda x:cv2.boundingRect(x)[0])
        # contours = sorted(contours, key=lambda x:cv2.boundingRect(x)[1])
        text = ''
        for ctr in contours:
            x,y,w,h = cv2.boundingRect(ctr)
            area = w*h
            ratio = w/h
            if area < 5000 or area > 16000 or ratio > 0.8: # 
                continue
            roi = image[y-5:y+h+5, x-5:x+w+5]
            roi = cv2.bitwise_not(roi)
            image = cv2.GaussianBlur(image, (5, 5), 0)
            res, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
            cv2.imshow(str(ratio), roi)
            print(x)        
            text_temp = (pytesseract.image_to_string(roi, config=' -l numberplate --psm 8 --oem 3'))
            text += (re.sub('[\W_]+', '', text_temp))    
                
        print("Text : {}".format(text))
        cv2.destroyAllWindows()
        return 1
    except:
        cv2.destroyAllWindows()
        return 1 

# ocr()
