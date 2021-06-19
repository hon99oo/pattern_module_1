import os
from random import *
import cv2
import numpy as np
import re

folders = ["견고딕", "굴림", "HY신명조"]

w = 500
lines = range(3,10,2) # 탐지할 선 갯수 목록

def getRectContour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 경계값 적용 cv.adaptiveThreshold(이미지, 최대값, 경계값 계산 알고리즘, 경계화 타입, 블록 크기(홀수), 보정상수)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)
   
    # 외곽선 찾기
    contours, hierarchy = cv2.findContours(
        thr,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # 외곽선 위치
    x, y, w, h = cv2.boundingRect(contours[0])

    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "from": (x, y),
        "to": (x+w, y+h),
        "contours": contours
    }

def loadImage(font_name, i):
    img_path = os.path.join(os.getcwd(), 'fonts', font_name, str(i) + '.png')
        
    img_array = np.fromfile(img_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    return img

def analyzeImage(font_name, i, line_cnt, show_window=False):
    img = loadImage(font_name, i) # 이미지 로딩

    rect = getRectContour(img) # 외곽선 위치 값 찾기

    cropped = img[rect["from"][1]:rect["to"][1], rect["from"][0]:rect["to"][0]] # 외곽선 위치로 이미지 자르기

    canvas = cv2.resize(cropped, (w, w), interpolation=cv2.INTER_AREA) # 자른 이미지 w 값 크기로 리사이징

    preview = canvas.copy() # 미리보기용 이미지
    
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) # 탐색할 이미지는 GRAYSCALE로 변환

    each = int(w/line_cnt) # 탐지선 간격 

    cnt = 0 # 탐지 횟수

    for j in range(line_cnt + 1):
        v = each * j

        if j == 0 or j == line_cnt:
            continue

        # 미리보기 이미지에 탐지선 긋기
        cv2.line(preview, (v, 0), (v, w), (0, 0, 255), 1)
        cv2.line(preview, (0, v), (w, v), (0, 0, 255), 1)

        started = False
        prev_filled = False
        for x in range(w):
            y = v

            if canvas[y, x] <= 255 / 3:
                if started is False:
                    started = True
                    cv2.circle(preview, (x, y), 2, (0, 255, 0), -1)
                    cnt = cnt + 1

                prev_filled = True
            else:
                if started is True and prev_filled is True:
                    cv2.circle(preview, (x, y), 2, (255, 0, 0), -1)
                    started = False

                prev_filled = False
        
        started = False
        prev_filled = False
        for y in range(w):
            x = v

            if canvas[y, x] <= 255 / 3:
                if started is False:
                    started = True
                    cv2.circle(preview, (x, y), 2, (0, 255, 0), -1)
                    cnt = cnt + 1

                prev_filled = True
            else:
                if started is True and prev_filled is True:
                    cv2.circle(preview, (x, y), 2, (255, 0, 0), -1)
                    started = False

                prev_filled = False

    if show_window:
        font_name = str(randint(1,100000))
        cv2.namedWindow(font_name + str(i) + str(line_cnt), cv2.WINDOW_NORMAL)
        cv2.imshow(font_name + str(i) + str(line_cnt), preview)

    return cnt


num_map = {}

#print("Width: " + str(w) + " / Lines: " + str(lines))

for i in range(10):
    print("Number: " + str(i))

    for k in range(3):
        font_name = folders[k]
        str_row = font_name + ' -'
        tmp = []
        for line_cnt in lines:
            
            cnt = analyzeImage(font_name, i, line_cnt, show_window=False)
            str_row = str_row + ' ' + str(cnt)
            tmp.append(cnt)

        num_map.setdefault(font_name + str(i), tmp)
        print(str_row)
    
    print("===========================")

# for key,items in num_map.items():
#     print(key,items)


#-----------------------------------------------------------------------------------------------------------------

def loadImage2(font_name, i):
    img_path = os.path.join(os.getcwd(), 'fonts', font_name, str(i) + '.png')
        
    img_array = np.fromfile(img_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    return img

def analyzeImage2(font_name, i, line_cnt, show_window=False):
    img = loadImage2(font_name, i) # 이미지 로딩

    rect = getRectContour(img) # 외곽선 위치 값 찾기

    cropped = img[rect["from"][1]:rect["to"][1], rect["from"][0]:rect["to"][0]] # 외곽선 위치로 이미지 자르기

    canvas = cv2.resize(cropped, (w, w), interpolation=cv2.INTER_AREA) # 자른 이미지 w 값 크기로 리사이징

    preview = canvas.copy() # 미리보기용 이미지
    
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) # 탐색할 이미지는 GRAYSCALE로 변환

    each = int(w/line_cnt) # 탐지선 간격 

    cnt = 0 # 탐지 횟수

    for j in range(line_cnt + 1):
        v = each * j

        if j == 0 or j == line_cnt:
            continue

        # 미리보기 이미지에 탐지선 긋기
        cv2.line(preview, (v, 0), (v, w), (0, 0, 255), 1)
        cv2.line(preview, (0, v), (w, v), (0, 0, 255), 1)

        started = False
        prev_filled = False
        for x in range(w):
            y = v

            if canvas[y, x] <= 255 / 3:
                if started is False:
                    started = True
                    cv2.circle(preview, (x, y), 2, (0, 255, 0), -1)
                    cnt = cnt + 1

                prev_filled = True
            else:
                if started is True and prev_filled is True:
                    cv2.circle(preview, (x, y), 2, (255, 0, 0), -1)
                    started = False

                prev_filled = False
        
        started = False
        prev_filled = False
        for y in range(w):
            x = v

            if canvas[y, x] <= 255 / 3:
                if started is False:
                    started = True
                    cv2.circle(preview, (x, y), 2, (0, 255, 0), -1)
                    cnt = cnt + 1

                prev_filled = True
            else:
                if started is True and prev_filled is True:
                    cv2.circle(preview, (x, y), 2, (255, 0, 0), -1)
                    started = False

                prev_filled = False

    if show_window:
        font_name = str(randint(1,100000))
        cv2.namedWindow(font_name + str(i) + str(line_cnt), cv2.WINDOW_NORMAL)
        cv2.imshow(font_name + str(i) + str(line_cnt), preview)

    return cnt

answers = ["0", "1", "9", "8", "4", "3", "2", "5", "6", "7"]
answers2 = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",]

suc_cnt = 0
for idx in range(10):

    font_name = "test2"
    checking = []
    for line_cnt in lines:
        cnt = analyzeImage2(font_name, idx, line_cnt, show_window=False)
        checking.append(cnt)

    #print(checking)
    maxN = 0
    order = 0
    count = 0
    cnt = 0
    else_count = 0
    min_else_count = 10000
    for key, values in num_map.items():
        print("START!!!!")
        for i,j in zip(values,checking):
            print(key,i, j)
            if i==j:
                count += 1
            else :
                else_count = else_count + abs(i-j)
        print("STOP!!!!")
        if order >= 2:
            if count > maxN:
                if else_count < min_else_count:
                    maxN = count
                    predict = key
                    max_else_count = else_count
            print("Count : ",count)
            print("ElseCount : ",else_count)
            count = 0
            else_count = 0
            order = -1
        order += 1



    accuracy = str(maxN/len(lines)*100) + "%"
    # print("P_list : ", checking)
    # print("A_list : ", num_map[predict])
    print("predicted : " + predict)
    print("answer : " + answers2[idx])
    if int(answers2[idx])== int(re.findall("\d",predict)[0]):
        print("SUCCESS")
        suc_cnt= suc_cnt+1
    else :
        print("FAIL")
    print("accuracy : " + accuracy)
    print("===============================")

print("Success is : ",suc_cnt,"/10")

cv2.waitKey(0)
cv2.destroyAllWindows()