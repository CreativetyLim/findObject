import cv2 as cv
import numpy as np
import math


color1 = 0
color2 = 0
ranges = 20
set_color = False
step = 0


def getAngle(startX,startY,endX, endY):

    dx =endX - startX
    dy =endY - startY

    return (math.atan2(dy, dx) * (180.0 / math.pi))+180


def nothing(x):
    global color1, color2
    global lower_blueA1
    global upper_blueA1
    global lower_blueB1
    global upper_blueB1
    global centerAX,centerAY, pointX, pointY
    global centerBX,centerBY 

    saturation_th1 = cv.getTrackbarPos('saturation_th1', 'img_result')
    value_th1 = cv.getTrackbarPos('value_th1', 'img_result')

    saturation_th2 = cv.getTrackbarPos('saturation_th2', 'img_result')
    value_th2 = cv.getTrackbarPos('value_th2', 'img_result')

    color1 = int(color1)
    color2 = int(color2)

    # HSV 색공간에서 마우스 클릭으로 얻은 픽셀값과 유사한 필셀값의 범위를 정합니다.
    if color1 < ranges:
        lower_blueA1 = np.array([color1 - ranges + 180, saturation_th1, value_th1])
        upper_blueA1 = np.array([180, 255, 255])
        

    elif color1 > 180 - ranges:
        lower_blueA1 = np.array([color1, saturation_th1, value_th1])
        upper_blueA1 = np.array([180, 255, 255])
        
    else:
        lower_blueA1 = np.array([color1, saturation_th1, value_th1])
        upper_blueA1 = np.array([color1 + ranges, 255, 255])


    if color2 < ranges:
        lower_blueB1 = np.array([color2 - ranges + 180, saturation_th2, value_th2])
        upper_blueB1 = np.array([180, 255, 255])

    elif color2 > 180 - ranges:
        lower_blueB1 = np.array([color2, saturation_th2, value_th2])
        upper_blueB1 = np.array([180, 255, 255])

    else:
        lower_blueB1 = np.array([color2, saturation_th2, value_th2])
        upper_blueB1 = np.array([color2 + ranges, 255, 255])
       
cv.namedWindow('img_color')
cv.namedWindow('img_result')

cv.createTrackbar('saturation_th1', 'img_result', 0, 255, nothing)
cv.setTrackbarPos('saturation_th1', 'img_result', 30)
cv.createTrackbar('value_th1', 'img_result', 0, 255, nothing)
cv.setTrackbarPos('value_th1', 'img_result', 30)
cv.createTrackbar('saturation_th2', 'img_result', 0, 255, nothing)
cv.setTrackbarPos('saturation_th2', 'img_result', 30)
cv.createTrackbar('value_th2', 'img_result', 0, 255, nothing)
cv.setTrackbarPos('value_th2', 'img_result', 30)

cap = cv.VideoCapture(0)

while(True):

    ret,img_color = cap.read()
    img_color = cv.flip(img_color, 1)

    if ret == False:
        continue


    img_color2 = img_color.copy()
    img_hsv = cv.cvtColor(img_color2, cv.COLOR_BGR2HSV)

    height, width = img_color.shape[:2]
    cx = int(width / 2)
    cy = int(height / 2)


    if set_color == False:

        rectangle_color = (0, 255, 0)

        if step == 1:
            rectangle_color = (0, 0, 255)

        cv.rectangle(img_color, (cx - 20, cy - 20), (cx + 20, cy + 20), rectangle_color, 5)


    else:

        # 범위 값으로 HSV 이미지에서 마스크를 생성합니다.
        img_maskA1 = cv.inRange(img_hsv, lower_blueA1, upper_blueA1)
        img_maskA = img_maskA1

        img_maskB1 = cv.inRange(img_hsv, lower_blueB1, upper_blueB1)
        img_maskB = img_maskB1


        # 모폴로지 연산
        kernel = np.ones((11,11), np.uint8)
        img_maskA = cv.morphologyEx(img_maskA, cv.MORPH_OPEN, kernel)
        img_maskA = cv.morphologyEx(img_maskA, cv.MORPH_CLOSE, kernel)


        kernel = np.ones((11,11), np.uint8)
        img_maskB = cv.morphologyEx(img_maskB, cv.MORPH_OPEN, kernel)
        img_maskB = cv.morphologyEx(img_maskB, cv.MORPH_CLOSE, kernel)


        # 마스크 이미지로 원본 이미지에서 범위값에 해당되는 영상 부분을 획득합니다.
        img_maskC = cv.bitwise_or(img_maskA, img_maskB)
        img_result = cv.bitwise_and(img_color, img_color, mask=img_maskC)


        # 라벨링
        numOfLabelsA, img_labelA, statsA, centroidsA = cv.connectedComponentsWithStats(img_maskA)
        

        for idx, centroid in enumerate(centroidsA):
            if statsA[idx][0] == 0 and statsA[idx][1] == 0:
                continue

            if np.any(np.isnan(centroid)):
                continue

            x,y,width,height,area = statsA[idx]
            centerX,centerY = int(centroid[0]), int(centroid[1])
            centerAX = centerX
            centerAY = centerY
            pointX = (x+centerX)/2
            pointY = (y+centerY)/2

            if area > 1500:
                cv.circle(img_color, (centerX, centerY), 10, (0,0,255), 10)
                cv.rectangle(img_color, (x,y), (x+width,y+height), (0,0,255))


        numOfLabelsB, img_labelB, statsB, centroidsB = cv.connectedComponentsWithStats(img_maskB)
        for idx, centroid in enumerate(centroidsB):
            if statsB[idx][0] == 0 and statsB[idx][1] == 0:
                continue

            if np.any(np.isnan(centroid)):
                continue

            x,y,width,height,area = statsB[idx]
            centerX,centerY = int(centroid[0]), int(centroid[1])
            centerBX = centerX
            centerBY = centerY

            if area > 1500:
                cv.circle(img_color, (centerX, centerY), 10, (255,0,0), 10)
                #cv.rectangle(img_color, (x,y), (x+width,y+height), (255,0,0))

           
        
            cv.arrowedLine(img_color, (centerAX, centerAY), (centerBX, centerBY),(0,255,255) , 3)
            
           
        cnts = cv.findContours(img_maskA.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[-2]
        center = None
        if len(cnts) > 0:
            c = max(cnts, key=cv.contourArea)
            ((x, y), radius) = cv.minEnclosingCircle(c)
            M = cv.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            
        cv.putText(img_color,"("+str(center[0])+","+str(center[1])+")"+"angle:"+str(getAngle(centerAX, centerAY, centerBX, centerBY)), (center[0]+10,center[1]+15), cv.FONT_HERSHEY_SIMPLEX, 0.8,(0, 0, 255),1)
        cv.imshow('img_result', img_result)


    cv.imshow('img_color', img_color)


    key = cv.waitKey(1) & 0xFF

    if key == 27: # esc
        break

    elif key == 32: # space
        if step == 0:
            roi = img_color2[cy-20:cy+20, cx-20:cx+20]
            roi = cv.medianBlur(roi, 3)
            cv.imshow("roi1", roi)
            hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
            h,s,v = cv.split(hsv)
            color1 = h.mean()
            print(color1)
            step += 1

        elif step == 1:
            roi = img_color2[cy-20:cy+20, cx-20:cx+20]
            roi = cv.medianBlur(roi, 3)
            cv.imshow("roi2", roi)
            hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
            h,s,v = cv.split(hsv)
            color2 = h.mean()
            set_color = True
            nothing(0)
            print(color2)
            step += 1

cap.release()
cv.destroyAllWindows()