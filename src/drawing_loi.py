import cv2

video_path = r"C:\Users\HRITHIK S\MY PROJECTS\footfall-counter\data\virat_dataset\virat_school_04.avi"

# Load first frame
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    raise SystemExit("❌ Failed to read video frame")

print(f"Frame resolution: {frame.shape[1]}x{frame.shape[0]}")

# Create resizable window with full resolution
window_name = "Draw Line of Interest (Press Q to quit)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, frame.shape[1], frame.shape[0])

points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Clicked Point: {x, y}")

        if len(points) == 2:
            cv2.line(frame, points[0], points[1], (255, 0, 0), 2)
            cv2.imshow(window_name, frame)
            print(f"✅ Final Line Coordinates: {points}")
            print(f"YAML Format: line: {points}")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

cv2.imshow(window_name, frame)
cv2.setMouseCallback(window_name, click_event)

print("👉 Click two points to draw your Line of Interest (LOI)")
print("Press 'Q' to quit anytime")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or len(points) == 2:
        break

cv2.destroyAllWindows()
