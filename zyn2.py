import cv2
import time
import numpy as np

def preprocess_frame(frame):
    """Preprocess frame to improve QR code detection and focus on a Zyn can."""
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
    """Estimate frame sharpness in the region of interest using Laplacian variance."""
    if roi is not None:
        gray = gray[roi[1]:roi[3], roi[0]:roi[2]]
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    return sharpness

def get_roi(frame):
    """Define a centered region of interest for QR code detection."""
    h, w = frame.shape[:2]
    roi_size = min(w, h) // 3
    x1 = (w - roi_size) // 2
    y1 = (h - roi_size) // 2
    x2 = x1 + roi_size
    y2 = y1 + roi_size
    return (x1, y1, x2, y2)

def rotate_image(image, angle):
    """Rotate the image by the specified angle (degrees)."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated, M

def correct_perspective(image, points):
    """Correct perspective distortion using the QR code's corner points."""
    if points is None or len(points) < 4:
        return image, None
    
    # Get the four corners of the QR code
    pts = points.reshape(4, 2)
    
    # Define the destination points (a square)
    side = max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[1] - pts[2]))
    dst_pts = np.array([
        [0, 0],
        [side, 0],
        [side, side],
        [0, side]
    ], dtype=np.float32)
    
    # Compute perspective transform
    M = cv2.getPerspectiveTransform(pts, dst_pts)
    corrected = cv2.warpPerspective(image, M, (int(side), int(side)))
    return corrected, M

# Try different camera indices
cam = None
for index in range(3):
    cam = cv2.VideoCapture(index)
    if cam.isOpened():
        print(f"Successfully opened webcam at index {index}")
        break
    cam.release()
else:
    print("Error: Could not open any webcam. Check connections, drivers, or permissions.")
    exit()

# Set resolution
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cam.set(cv2.CAP_PROP_FPS, 60)

# Check actual resolution
actual_width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Using resolution: {actual_width}x{actual_height}")

# Initialize QR code detector
qr_detector = cv2.QRCodeDetector()

# Set to track unique codes
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
        # Try detecting QR code at different rotations
        code = None
        points = None
        for angle in [0, 90, 180, 270]:
            # Rotate the ROI
            rotated_roi, M = rotate_image(processed_roi, angle)
            
            # Detect QR code in rotated ROI
            code, points, _ = qr_detector.detectAndDecode(rotated_roi)
            if code:
                # Rotate points back to original frame coordinates
                if points is not None and len(points) >= 4:
                    points = points.reshape(-1, 1, 2)
                    M_inv = cv2.invertAffineTransform(M)
                    points = cv2.transform(points, M_inv)
                    points = points.reshape(4, 2)
                    # Adjust points to full frame coordinates
                    points += np.array([x1, y1])
                break
        
        # If still not detected, try perspective correction on the original ROI
        if not code:
            code, points, _ = qr_detector.detectAndDecode(processed_roi)
            if code and points is not None:
                corrected_roi, M = correct_perspective(processed_roi, points)
                if corrected_roi is not None:
                    code, corrected_points, _ = qr_detector.detectAndDecode(corrected_roi)
                    if code and corrected_points is not None:
                        # Transform points back to original frame coordinates
                        corrected_points = corrected_points.reshape(-1, 1, 2)
                        M_inv = np.linalg.inv(M)
                        corrected_points = cv2.perspectiveTransform(corrected_points, M_inv)
                        corrected_points = corrected_points.reshape(4, 2)
                        points = corrected_points + np.array([x1, y1])
        
        if code and code not in codes:
            print(f"QR Code Detected: {code}")
            status_text = "QR Code Detected"
            
            codes.add(code)
            with open("zyn_codes.txt", "a") as f:
                f.write(f"{code}\n")
            print(f"Saved to zyn_codes.txt")
            
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

cam.release()
cv2.destroyAllWindows()
print("Stopped scanning.")