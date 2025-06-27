import cv2
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

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

    # Try different backends to avoid MSMF issues
    backends = [cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_ANY]
    cam = None
    
    for backend in backends:
        for index in range(3):
            try:
                cam = cv2.VideoCapture(index, backend)
                if cam.isOpened():
                    # Test if we can actually read a frame
                    ret, test_frame = cam.read()
                    if ret and test_frame is not None:
                        print(f"Successfully opened webcam {index} with backend {backend}")
                        break
                    else:
                        cam.release()
                        cam = None
            except:
                if cam:
                    cam.release()
                cam = None
        if cam and cam.isOpened():
            break
    
    if not cam or not cam.isOpened():
        print("Error: Could not open any webcam with any backend.")
        exit()
    
    print(f"Successfully opened webcam")

    # Set camera properties with lower settings to avoid MSMF issues
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_FPS, 30)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to avoid frame delays

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
        
        status_text = "Scanning... Adjust camera if needed."
        current_time = time.time()
        
        if current_time - last_detection_time >= detection_cooldown:
            # Try multiple preprocessing approaches for better QR detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Try original frame first
            code, points, _ = qr_detector.detectAndDecode(frame)
            
            # If not found, try grayscale
            if not code:
                code, points, _ = qr_detector.detectAndDecode(gray)
            
            # If still not found, try with contrast enhancement
            if not code:
                enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
                code, points, _ = qr_detector.detectAndDecode(enhanced)
            
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
                
                last_detection_time = current_time
        
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