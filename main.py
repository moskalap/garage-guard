import numpy as np
import cv2
import imutils
import pytesseract

allowed_car_plates = [line.rstrip('\n') for line in open('allowed_car_plates')]

def crop_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 170, 200)
    (new, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    contour = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            contour = approx
            break
    return contour


def extract_car_plate_text(img, contour):
    mask = np.zeros(img.shape, dtype=np.uint8)
    roi_corners = np.array([contour], dtype=np.int32)
    channel_count = img.shape[2]
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    car_plate = cv2.bitwise_and(img, mask)
    text = pytesseract.image_to_string(car_plate)
    print(text)
    return text


def handle_text(text):
    if str(text).capitalize() in allowed_car_plates:
        print('opening gate for ' + text)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("no camera connected")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    image = imutils.resize(frame, width=500)
    plate_countour = crop_plate(image)
    if plate_countour:
        cv2.drawContours(image, [plate_countour], -1, (0, 255, 0), 3)
        text = extract_car_plate_text(plate_countour, image)
        handle_text(text)
cap.release()
cv2.destroyAllWindows()


