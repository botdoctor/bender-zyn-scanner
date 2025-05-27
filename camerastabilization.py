import cv2
import time
import numpy as np
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    alpha = 4.0
    beta = -120
    adjusted = cv2.convertScaleAbs(blurred, alpha=alpha, beta=beta)
    thresh = cv2.adaptiveThreshold(
        adjusted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(thresh, -1, kernel)
    return sharpened, gray

def estimate_sharpness(gray, roi=None):
    if roi is not None:
        gray = gray[roi[1]:roi[3], roi[0]:roi[2]]
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    return sharpness

def get_roi(frame):
    h, w = frame.shape[:2]
    roi_size = min(w, h) // 3
    x1 = (w - roi_size) // 2
    y1 = (h - roi_size) // 2
    x2 = x1 + roi_size
    y2 = y1 + roi_size
    return (x1, y1, x2, y2)

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated, M

def correct_perspective(image, points):
    if points is None or len(points) < 4:
        return image, None
    pts = points.reshape(4, 2)
    side = max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[1] - pts[2]))
    dst_pts = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst_pts)
    corrected = cv2.warpPerspective(image, M, (int(side), int(side)))
    return corrected, M

def extract_code_from_url(url):
    if not url or '/' not in url:
        return None
    return url.split('/')[-1]

def paste_and_submit_code(driver, code):
    try:
        # Wait for the code field to be present on the current page
        code_field = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.ID, "zyn-rewards-select"))
        )

        # Simulate human typing with minimal delay
        code_field.clear()
        for char in code:
            code_field.send_keys(char)
            time.sleep(0.01)  # Minimal delay per character
        time.sleep(0.5)  # Short pause after typing

        # Press Enter to submit the form
        code_field.send_keys(Keys.ENTER)
        time.sleep(1)  # Short pause after submission

        print(f"Pasted and submitted code {code}. Please handle any CAPTCHAs that appear.")
    except Exception as e:
        print(f"Error pasting and submitting code {code}: {e}")

# Initialize Selenium WebDriver with remote debugging
chrome_options = Options()
# Connect to the existing Chrome session via remote debugging port
chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
# Minimize automation flags
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
# Set a user agent to mimic a real browser
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# Define cam as None initially to avoid NameError
cam = None

try:
    # Assume the user is already on the rewards page with remote debugging
    print("Please ensure Chrome is open with remote debugging (port 9222) and you are on https://us.zyn.com/en_us/zynrewards.")
    time.sleep(3)

    print("Proceeding to scan QR codes, paste, and submit them on the current page.")

    for index in range(3):
        cam = cv2.VideoCapture(index)
        if cam.isOpened():
            print(f"Successfully opened webcam at index {index}")
            break
        cam.release()
    else:
        print("Error: Could not open any webcam. Check connections, drivers, or permissions.")
        exit()

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cam.set(cv2.CAP_PROP_FPS, 60)

    actual_width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Using resolution: {actual_width}x{actual_height}")

    qr_detector = cv2.QRCodeDetector()
    codes = set()

    print("Scanning for QR codes on Zyn can. Press 'q' to quit.")

    last_detection_time = 0
    detection_cooldown = 3

    while True:
        success, frame = cam.read()
        if not success or frame is None:
            print("Error: Failed to capture frame.")
            break
        
        roi = get_roi(frame)
        x1, y1, x2, y2 = roi
        
        processed, gray = preprocess_frame(frame)
        processed_roi = processed[y1:y2, x1:x2]
        
        sharpness = estimate_sharpness(gray, roi)
        focus_tip = "Move slowly 3-6 inches away, tilt to avoid glare."
        if sharpness < 50:
            focus_tip = "Move closer (3-5 inches) or adjust lighting."
        elif sharpness > 150:
            focus_tip = "Move farther (5-8 inches) or adjust angle."
        
        status_text = f"Scanning... {focus_tip}"
        current_time = time.time()
        
        if current_time - last_detection_time >= detection_cooldown:
            code = None
            points = None
            for angle in [0, 90, 180, 270]:
                rotated_roi, M = rotate_image(processed_roi, angle)
                code, points, _ = qr_detector.detectAndDecode(rotated_roi)
                if code:
                    if points is not None and len(points) >= 4:
                        points = points.reshape(-1, 1, 2)
                        M_inv = cv2.invertAffineTransform(M)
                        points = cv2.transform(points, M_inv)
                        points = points.reshape(4, 2)
                        points += np.array([x1, y1])
                    break
            
            if not code:
                code, points, _ = qr_detector.detectAndDecode(processed_roi)
                if code and points is not None:
                    corrected_roi, M = correct_perspective(processed_roi, points)
                    if corrected_roi is not None:
                        code, corrected_points, _ = qr_detector.detectAndDecode(corrected_roi)
                        if code and corrected_points is not None:
                            corrected_points = corrected_points.reshape(-1, 1, 2)
                            M_inv = np.linalg.inv(M)
                            corrected_points = cv2.perspectiveTransform(corrected_points, M_inv)
                            corrected_points = corrected_points.reshape(4, 2)
                            points = corrected_points + np.array([x1, y1])
            
            if code and code not in codes:
                print(f"QR Code Detected: {code}")
                status_text = "QR Code Detected"
                
                qr_code = extract_code_from_url(code)
                if qr_code:
                    codes.add(qr_code)
                    with open("zyn_codes.txt", "a") as f:
                        f.write(f"{qr_code}\n")
                    print(f"Saved to zyn_codes.txt")
                    
                    paste_and_submit_code(driver, qr_code)
                else:
                    print("Invalid URL format in QR code.")
                
                if points is not None and len(points) >= 4:
                    pts = points.astype(int)
                    for i in range(len(pts)):
                        cv2.line(frame, tuple(pts[i][0]), tuple(pts[(i+1)%len(pts)][0]), (0, 255, 0), 2)
                
                last_detection_time = current_time
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("QR Code Scanner", frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

finally:
    if cam is not None:
        cam.release()
    cv2.destroyAllWindows()
    driver.quit()
    print("Stopped scanning and closed browser.")