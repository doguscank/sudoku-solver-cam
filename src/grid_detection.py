import cv2
import numpy as np
import itertools
import matplotlib.pyplot as plt
import math

cv2.namedWindow('test', cv2.WINDOW_NORMAL)

def DetectGrid(img):
	img = cv2.resize(img, (512, 512))

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 5, 2)

	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	max_contour_area = 1000
	best_contour = None

	for contour in contours:
		area = cv2.contourArea(contour)

		if area > max_contour_area:
			max_contour_area = area
			best_contour = contour
	
	approx_poly = cv2.approxPolyDP(best_contour, 1, True)
	approx_poly = np.array([a[0] for a in approx_poly])

	combinations = np.array(list(itertools.combinations(approx_poly, 4)))

	best_combination = None
	max_area = 1000

	for c in combinations:
		area = cv2.contourArea(c)

		if area > max_area:
			max_area = area
			best_combination = c

	#print(best_combination)

	x_sorted = sorted(best_combination, key = lambda x: x[0])
	#print(x_sorted)

	p_1 = min(x_sorted[:2], key = lambda x: x[1])
	p_3 = max(x_sorted[:2], key = lambda x: x[1])
	p_2 = min(x_sorted[2:], key = lambda x: x[1])
	p_4 = max(x_sorted[2:], key = lambda x: x[1])
	#print(p_1, p_2, p_3, p_4)

	pts_1 = np.float32([p_1, p_2, p_3, p_4])
	#print(pts_1)

	pts_2 = np.float32([[0, 0], [512, 0], [0, 512], [512, 512]])
	#print(pts_2)

	mask = np.zeros(gray.shape, np.uint8)
	cv2.drawContours(mask, [best_combination], -1, 255, -1)
	cv2.drawContours(mask, [best_combination], 0, 0, 2)

	masked_img = cv2.bitwise_and(img, img, mask = mask)

	#for a in best_combination:
	#	cv2.circle(masked_img, (a[0], a[1]), 3, (0, 0, 255), -1)
	#	cv2.putText(masked_img, "{}, {}".format(a[0], a[1]), (a[0], a[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0))

	M = cv2.getPerspectiveTransform(pts_1, pts_2)
	dst = cv2.warpPerspective(masked_img, M, (512, 512))

	return dst

def DetectSubGrids(grid):
	gray = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 5, 2)
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (3, 3), iterations = 2)

	#contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	#for contour in contours:
	#	if cv2.contourArea(contour) > 4000:
	#		cv2.drawContours(grid, contour, -1, (0, 0, 255), -1)

	lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 150, 50, 30)

	print(len(lines))

	for line in lines:
		line = line[0]
		cv2.line(grid, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 1)

	return thresh

#Testing the functions here
if __name__ == '__main__':
	img = cv2.imread('./images/sudoku_newspaper_6.jpg')
	img = DetectGrid(img)
	cv2.imshow('test', img)
	cv2.waitKey(0)

	thresh = DetectSubGrids(img)
	cv2.imshow('test', thresh)
	cv2.waitKey(0)

	#for i in range(8):
	#	for j in range(8):
	#		cv2.imshow('test', img[i * 64: (i + 1) * 64 - 1, j * 64: (j + 1) * 64 - 1])
	#		cv2.waitKey(0)