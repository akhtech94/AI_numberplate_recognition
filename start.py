import ExtractString
import yolov4
import time

while True:
    yolov4.detect()
    time.sleep(1)
    ExtractString.ocr()
    print("Done")
