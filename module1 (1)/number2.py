import os
import sys
import json
from random import *
import cv2
import numpy as np

present_window = False
mode = 0

input_file = sys.argv[1]

if input_file is None:
    print("Invalid argument")
    sys.exit()

w = 100
lines = []

for i in range(2, w + 1):
    if w % i == 0:
        lines.append(i)

lines = lines[1: -3]


def get_rect_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 경계값 적용 cv.adaptiveThreshold(이미지, 최대값, 경계값 계산 알고리즘, 경계화 타입, 블록 크기(홀수), 보정상수)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)

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
        "to": (x + w, y + h),
        "contours": contours
    }


def load_image(font_name, i):
    img_path = os.path.join(os.getcwd(), 'fonts', font_name, str(i) + '.png')

    img_array = np.fromfile(img_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    return img


def load_image_by_path(path):
    img_path = path

    img_array = np.fromfile(img_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    return img


def analyze_image(img, line_cnt, show_window=False):
    rect = get_rect_contour(img)  # 외곽선 위치 값 찾기

    cropped = img[rect["from"][1]:rect["to"][1],
              rect["from"][0]:rect["to"][0]]  # 외곽선 위치로 이미지 자르기

    # 자른 이미지 w 값 크기로 리사이징
    canvas = cv2.resize(cropped, (w, w), interpolation=cv2.INTER_AREA)

    preview = canvas.copy()  # 미리보기용 이미지

    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)  # 탐색할 이미지는 GRAYSCALE로 변환

    each = int(w / line_cnt)  # 탐지선 간격

    v_cnt = 0  # 탐지 횟수 (가로)
    h_cnt = 0  # 탐지 횟수 (세로)

    for j in range(line_cnt + 1):
        v = each * j

        if j == 0 or j == line_cnt:
            continue

        # 미리보기 이미지에 탐지선 긋기
        cv2.line(preview, (v, 0), (v, w), (0, 0, 255), 1)
        # cv2.line(preview, (0, v), (w, v), (0, 0, 255), 1)

        started = False
        prev_filled = False
        for x in range(w):
            y = v

            if canvas[y, x] <= 240:
                if started is False:
                    started = True
                    cv2.circle(preview, (x, y), 2, (0, 255, 0), -1)
                    v_cnt = v_cnt + 1

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

            if canvas[y, x] <= 240:
                if started is False:
                    started = True
                    cv2.circle(preview, (x, y), 2, (0, 255, 0), -1)
                    h_cnt = h_cnt + 1

                prev_filled = True
            else:
                if started is True and prev_filled is True:
                    cv2.circle(preview, (x, y), 2, (255, 0, 0), -1)
                    started = False

                prev_filled = False

    if show_window:
        present_window = True
        font_name = str(randint(1, 100000))
        cv2.namedWindow(font_name + str(i) + str(line_cnt), cv2.WINDOW_NORMAL)
        cv2.imshow(font_name + str(i) + str(line_cnt), preview)

    return [v_cnt, h_cnt]


# print("Width: " + str(w) + " / Lines: " + str(lines))

num_map = {}

if (input_file == 'make'):
    folders = ["견고딕", "굴림", "HY신명조"]

    for i in range(10):
        print("Number: " + str(i))

        arr_line = []
        for line_cnt in lines:
            arr_font = []
            for k in range(3):
                font_name = folders[k]
                str_row = font_name + ' -'

                img = load_image(font_name, i)  # 이미지 로딩

                [v_cnt, h_cnt] = analyze_image(img, line_cnt, show_window=False)
                arr_font.append(h_cnt + v_cnt)

            arr_line.append(arr_font)

        print(arr_line)
        print("===========================")

        num_map[i] = arr_line

    with open('./data.json', 'w', encoding='utf-8') as f:
        json.dump(num_map, f)

    print('generated. check the data.json file')
else:
    num_map = {}

    with open('./data.json', 'r') as f:
        num_map = json.load(f)

    checking = []

    img = load_image_by_path(input_file)

    for line_cnt in lines:
        [v_cnt, h_cnt] = analyze_image(img, line_cnt, show_window=False)
        checking.append(h_cnt + v_cnt)

    percentages = []
    for row in num_map.values():
        matches = []
        for i, n in enumerate(checking):
            matches.append(n in row[i])

        cnt = 0
        for tf in matches:
            if tf:
                cnt += 1

        percentages.append(cnt / len(matches))

    print(percentages)
    max_percent = max(percentages)

    # if(max_percent < 0.6):
    #     print("cannot find number")
    #     sys.exit()

    max_idx = percentages.index(max_percent)
    print(max_idx, '/', max_percent * 100, '%')

    # print(cnt_list.index(max(cnt_list)), "/", max(cnt_list)/len(lines)*100, "%")

if present_window:
    cv2.waitKey(0)
    cv2.destroyAllWindows()