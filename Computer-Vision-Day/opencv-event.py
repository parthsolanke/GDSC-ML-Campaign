import cv2

# Read and display an image
image = cv2.imread('image.jpg')
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Read and display a video
video = cv2.VideoCapture('video.mp4')
while True:
    ret, frame = video.read()
    if not ret:
        break
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) == ord('q'):
        break
video.release()
cv2.destroyAllWindows()

# Read webcam video and display it
webcam = cv2.VideoCapture(0)
while True:
    ret, frame = webcam.read()
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == ord('q'):
        break
webcam.release()
cv2.destroyAllWindows()

# Resize an image
resized_image = cv2.resize(image, (100, 200))
cv2.imshow('Resized Image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Resize a video
output_width = 400
output_height = 400
output_video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (output_width, output_height))
while True:
    ret, frame = video.read()
    if not ret:
        break
    resized_frame = cv2.resize(frame, (output_width, output_height))
    output_video.write(resized_frame)
output_video.release()

# Apply Canny edge detection on an image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray_image, 100, 200)
cv2.imshow('Canny Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Draw on an image
cv2.rectangle(image, (100, 200), (150, 250), (0, 255, 0), 5)
cv2.circle(image, (300, 200), 60, (0, 0, 255), 5)
cv2.line(image, (70, 80), (90, 120), (255, 0, 0), 8)
cv2.imshow('Image with Drawings', image)
cv2.waitKey(0)
cv2.destroyAllWindows()