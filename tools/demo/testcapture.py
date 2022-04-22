import cv2

print(cv2.__version__)
cap = cv2.VideoCapture(r'D:\Projects\Python\YOLOX-main\tools\demo\test6.avi')
ret, frame = cap.read()
if not ret:
    print("1！！")
while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("test",frame)
        cv2.waitKey(0)
    else:

        break