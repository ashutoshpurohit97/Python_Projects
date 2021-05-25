import numpy as np
import cv2
import time


def draw_lines(img, prev_x1_l, prev_x2_l, prev_y1_l, prev_y2_l, prev_x1_r, prev_x2_r, prev_y1_r, prev_y2_r):
    image = img
    height, width = img.shape[:2]
    # print(height, width)  # 540, 960

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    kernel_size = 3
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    mask = np.zeros_like(edges)
    ignore_mask_color = 255

    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (480, 320), (490, 120), (imshape[1], imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 80  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 60  # minimum number of pixels making up a line
    max_line_gap = 150  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on

    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    sorted_lines = []
    slope_list = []
    left_list = []
    right_list = []
    # x1_list = x2_list = y1_list = y2_list = [10]

    threshold_slop = 0.5

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if(x2==x1):
            slope=(y2-y1)/0.8
        else:
            slope = (y2 - y1) / (x2 - x1)

        if abs(slope) > threshold_slop:
            slope_list.append(slope)
            sorted_lines.append(line)

            # In Comp Science, top left corner is origin so we are dealing with values in 4th quadrant.
            if slope < 0:
                right_list.append(line)
            elif slope > 0:
                left_list.append(line)

    for line in right_list:
        for x1, y1, x2, y2 in right_list[0]:
            if len(right_list) == 0:
                cv2.line(line_image, (prev_x1_l, prev_y1_l), (prev_x2_l, prev_y2_l), (0, 0, 255), 6)
            else:
                # cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
                prev_x1_l = x1
                prev_x2_l = x2
                prev_y1_l = y1
                prev_y2_l = y2

    for line in left_list:
        for x1, y1, x2, y2 in left_list[0]:
            if len(left_list) == 0:
                cv2.line(line_image, (prev_x1_r, prev_y1_r), (prev_x2_r, prev_y2_r), (0, 0, 255), 6)
            else:
                # cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
                prev_x1_r = x1
                prev_x2_r = x2
                prev_y1_r = y1
                prev_y2_r = y2

    color_edges = np.dstack((edges, edges, edges))

    annotated = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    del slope_list[:]

    return annotated, prev_x1_l, prev_x2_l, prev_y1_l, prev_y2_l, prev_x1_r, prev_x2_r, prev_y1_r, prev_y2_r


def cal_x_left(x1, x2, y1, y2):
    y_top = 600
    y_bottom = 450

    x_top = (y1 * (x1 - x2)) + (x1 * (y2 - y1)) - (y_top * (x1 - x2))
    x_top = x_top / (y2 - y1)

    x_bottom = (y1 * (x1 - x2)) + (x1 * (y2 - y1)) - (y_bottom * (x1 - x2))
    x_bottom = x_bottom / (y2 - y1)

    return x_top, x_bottom, y_top, y_bottom


# There is no need of this function. You can use above function only but for better understanding, I have written it separately
def cal_x_right(x1, x2, y1, y2):
    y_top = 600
    y_bottom = 450

    x_top = (y1 * (x1 - x2)) + (x1 * (y2 - y1)) - (y_top * (x1 - x2))
    x_top = x_top / (y2 - y1)

    x_bottom = (y1 * (x1 - x2)) + (x1 * (y2 - y1)) - (y_bottom * (x1 - x2))
    x_bottom = x_bottom / (y2 - y1)

    return x_top, x_bottom, y_top, y_bottom


cap = cv2.VideoCapture('video2.mp4')  # Set the path of the video here.

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter('Augumented_result2.avi', fourcc, 20.0, (1280,720 ))

prev_x1_l, prev_x2_l, prev_y1_l, prev_y2_l, prev_x1_r, prev_x2_r, prev_y1_r, prev_y2_r = 0, 0, 0, 0, 0, 0, 0, 0

while (cap.isOpened()):
    ret, img = cap.read()
    #     print(img.shape[:2])

    if ret == 0:
        break
    else:
        annotated, prev_x1_l, prev_x2_l, prev_y1_l, prev_y2_l, prev_x1_r, prev_x2_r, prev_y1_r, prev_y2_r = draw_lines(
            img, prev_x1_l, prev_x2_l, prev_y1_l, prev_y2_l, prev_x1_r, prev_x2_r, prev_y1_r, prev_y2_r)

        left_top_x, left_bottom_x, left_top_y, left_bottom_y = cal_x_left(prev_x1_l, prev_x2_l, prev_y1_l, prev_y2_l)
        right_top_x, right_bottom_x, right_top_y, right_bottom_y = cal_x_right(prev_x1_r, prev_x2_r, prev_y1_r,
                                                                               prev_y2_r)

        cv2.line(annotated, (int(left_top_x), int(left_top_y)), (int(left_bottom_x), int(left_bottom_y)), (0, 255, 0),
                 5)
        cv2.line(annotated, (int(right_top_x), int(right_top_y)), (int(right_bottom_x), int(right_bottom_y)),
                 (0, 255, 0), 5)

        # cv2.circle(annotated, (int(left_top_x), int(left_top_y)), 5, (255, 255, 0))
        # cv2.circle(annotated, (int(right_top_x), int(right_top_y)), 5, (255, 255, 0))
        # cv2.circle(annotated, (int(left_bottom_x), int(left_bottom_y)), 5, (0, 255, 255))
        # cv2.circle(annotated, (int(right_bottom_x), int(right_bottom_y)), 5, (0, 255, 255))

        # Offset calculation
        # Use top values for calculations coz we have to steer steering based on road ahead and not on road near to the car.
        lane_width_top = int(right_top_x - left_top_x)

        lane_center = (lane_width_top / 2) + left_top_x

        vehicle_center = 150+annotated.shape[1] / 2

        offset = int(vehicle_center - lane_center)

        # Display centers:
        cv2.circle(annotated, (int(lane_center), int(left_top_y)), 5, (255, 255, 0), 2)  # cyan = lane center
        cv2.circle(annotated, (int(vehicle_center), int(right_top_y)), 5, (0, 0, 255), 2)  # red = vehicle center

        # Indexing Part
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(annotated, 'Offset(in pixels): ', (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, str(offset), (280, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(annotated, 'Vehicle Center:', (10, 75), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(annotated, (140, 70), 5, (0, 0, 255), 2)  # red = vehicle center
        
        if(-100<=(offset)<=100):
            cv2.putText(annotated, 'Going straight',(500,250 ), font, 1, (0,255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(annotated, 'Changing the lane',(500,250 ), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
        cv2.putText(annotated, 'Lane Center:', (10, 95), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(annotated, (140, 90), 5, (255, 255, 0), 2)  # cyan = lane center

        out.write(annotated)
        cv2.imshow('Result', annotated)

    

    if cv2.waitKey(30) == 27:
        cap.release()
        break

cv2.destroyAllWindows()
