import numpy as np
import cv2
import time
import numpy as np
import math
from shapely.geometry import Point, LineString
import shapely

def cam2gray(cam):
    success, image = cam.read()
    img_g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return success, img_g

def getThreshold(cam, t):
    success, t_plus = cam2gray(cam)
    if not success:
        return None
    
    dimg = cv2.absdiff(t, t_plus)
    blur = cv2.GaussianBlur(dimg, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)

    # Define a kernel for the morphological operation
    kernel = np.ones((5, 5), np.uint8)

    # Closing to close small holes in the foreground
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Opening to remove noise
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    
    return opening



def diff2blur(cam, t):
    _, t_plus = cam2gray(cam)
    dimg = cv2.absdiff(t, t_plus)
    kernel = np.ones((5, 5), np.float32) / 25
    blur = cv2.filter2D(dimg, -1, kernel)
    return t_plus, blur

def getCorners(img_in):
    edges = cv2.goodFeaturesToTrack(img_in, 640, 0.0008, 1, mask=None, blockSize=3, useHarrisDetector=1, k=0.06)
    corners = np.intp(edges)
    return corners

def filterCorners(corners):
    mean_corners = np.mean(corners, axis=0)
    corners_new = np.array([i for i in corners if abs(mean_corners[0][0] - i[0][0]) <= 180 and abs(mean_corners[0][1] - i[0][1]) <= 120])
    return corners_new

def filterCornersLine(corners, rows, cols):
    [vx, vy, x, y] = cv2.fitLine(corners, cv2.DIST_HUBER, 0, 0.1, 0.1)
    lefty = int((-x[0] * vy[0] / vx[0]) + y[0])
    righty = int(((cols - x[0]) * vy[0] / vx[0]) + y[0])
    corners_final = np.array([i for i in corners if abs((righty - lefty) * i[0][0] - (cols - 1) * i[0][1] + cols * lefty - righty) / np.sqrt((righty - lefty)**2 + (cols - 1)**2) <= 40])
    return corners_final

class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        self.dt = dt
        self.u_x = u_x
        self.u_y = u_y
        self.std_acc = std_acc

        self.A = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        self.B = np.array([[(self.dt**2)/2, 0],
                           [0, (self.dt**2)/2],
                           [self.dt, 0],
                           [0, self.dt]])

        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        self.Q = np.array([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                           [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                           [(self.dt**3)/2, 0, self.dt**2, 0],
                           [0, (self.dt**3)/2, 0, self.dt**2]]) * self.std_acc**2

        self.R = np.array([[x_std_meas**2, 0],
                           [0, y_std_meas**2]])

        self.P = np.eye(4)
        self.x = np.zeros((4, 1))

    def predict(self):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, np.array([[self.u_x], [self.u_y]]))
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.H.shape[1])
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P),
                        (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

def getRealLocation(corners_final, mount, prev_tip_point=None, blur=None, kalman_filter=None):
    if mount == "right":
        loc = np.argmax(corners_final, axis=0)
    else:
        loc = np.argmin(corners_final, axis=0)
    locationofdart = corners_final[loc]
    
    # Skeletonize the dart contour
    dart_contour = corners_final.reshape((-1, 1, 2))
    skeleton = cv2.ximgproc.thinning(cv2.drawContours(np.zeros_like(blur), [dart_contour], -1, 255, thickness=cv2.FILLED))
    
    # Detect the dart tip using skeletonization and Kalman filter
    dart_tip = find_dart_tip(skeleton, prev_tip_point, kalman_filter)
    
    if dart_tip is not None:
        tip_x, tip_y = dart_tip
        # Draw a circle around the dart tip
        if blur is not None:
            cv2.circle(blur, (tip_x, tip_y), 1, (0, 255, 0), 1)
        
        locationofdart = dart_tip
    
    return locationofdart, dart_tip

from shapely.geometry import Point, LineString, Polygon

def find_dart_tip(skeleton, prev_tip_point, kalman_filter):
    # Find the contour of the skeleton
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # Find the contour with the maximum area (assuming it represents the dart)
        dart_contour = max(contours, key=cv2.contourArea)
        
        # Convert the contour to a Shapely Polygon
        dart_polygon = Polygon(dart_contour.reshape(-1, 2))
        
        # Find the lowest point of the dart contour
        dart_points = dart_polygon.exterior.coords
        lowest_point = max(dart_points, key=lambda x: x[1])
        
        # Consider the lowest point as the dart tip
        tip_point = lowest_point
        
        # Predict the dart tip position using the Kalman filter
        predicted_tip = kalman_filter.predict()
        
        # Update the Kalman filter with the observed dart tip position
        kalman_filter.update(np.array([[tip_point[0]], [tip_point[1]]]))
        
        return int(tip_point[0]), int(tip_point[1])
    
    return None


def main():
    cam_R = cv2.VideoCapture(0)  # Use the appropriate camera index for the right camera
    cam_L = cv2.VideoCapture(1)  # Use the appropriate camera index for the left camera
    cam_C = cv2.VideoCapture(2)  # Use the appropriate camera index for the center camera

    # Check if the cameras are opened successfully
    if not cam_R.isOpened() or not cam_L.isOpened() or not cam_C.isOpened():
        print("Failed to open one or more cameras.")
        return

    # Read first image twice to start loop
    _, _ = cam2gray(cam_R)
    _, _ = cam2gray(cam_L)
    _, _ = cam2gray(cam_C)
    time.sleep(0.1)
    success, t_R = cam2gray(cam_R)
    _, t_L = cam2gray(cam_L)
    _, t_C = cam2gray(cam_C)

    prev_tip_point_R = None
    prev_tip_point_L = None
    prev_tip_point_C = None
    
    # Initialize Kalman filters for each camera
    dt = 1.0 / 30.0  # Assuming 30 FPS
    u_x = 0
    u_y = 0
    std_acc = 1.0
    x_std_meas = 0.1
    y_std_meas = 0.1
    
    kalman_filter_R = KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
    kalman_filter_L = KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
    kalman_filter_C = KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)

    takeout_threshold = 20000  # Adjust this value based on the size of your hand and distance from the camera
    takeout_delay = 1.0  # Delay in seconds after takeout procedure

    while success:
        time.sleep(0.1)
        thresh_R = getThreshold(cam_R, t_R)
        thresh_L = getThreshold(cam_L, t_L)
        thresh_C = getThreshold(cam_C, t_C)

        if (cv2.countNonZero(thresh_R) > 1000 and cv2.countNonZero(thresh_R) < 7500) or (cv2.countNonZero(thresh_L) > 1000 and cv2.countNonZero(thresh_L) < 7500) or (cv2.countNonZero(thresh_C) > 1000 and cv2.countNonZero(thresh_C) < 7500):
            time.sleep(0.2)
            t_plus_R, blur_R = diff2blur(cam_R, t_R)
            t_plus_L, blur_L = diff2blur(cam_L, t_L)
            t_plus_C, blur_C = diff2blur(cam_C, t_C)
            corners_R = getCorners(blur_R)
            corners_L = getCorners(blur_L)
            corners_C = getCorners(blur_C)

            if corners_R.size < 40 and corners_L.size < 40 and corners_C.size < 40:
                print("### dart not detected")
                continue

            corners_f_R = filterCorners(corners_R)
            corners_f_L = filterCorners(corners_L)
            corners_f_C = filterCorners(corners_C)

            if corners_f_R.size < 30 and corners_f_L.size < 30 and corners_f_C.size < 30:
                print("### dart not detected")
                continue

            rows, cols = blur_R.shape[:2]
            corners_final_R = filterCornersLine(corners_f_R, rows, cols)
            corners_final_L = filterCornersLine(corners_f_L, rows, cols)
            corners_final_C = filterCornersLine(corners_f_C, rows, cols)

            _, thresh_R = cv2.threshold(blur_R, 60, 255, 0)
            _, thresh_L = cv2.threshold(blur_L, 60, 255, 0)
            _, thresh_C = cv2.threshold(blur_C, 60, 255, 0)

            if cv2.countNonZero(thresh_R) > 15000 or cv2.countNonZero(thresh_L) > 15000 or cv2.countNonZero(thresh_C) > 15000:
                continue

            print("Dart detected")

            try:
                locationofdart_R, prev_tip_point_R = getRealLocation(corners_final_R, "right", prev_tip_point_R, blur_R, kalman_filter_R)
                locationofdart_L, prev_tip_point_L = getRealLocation(corners_final_L, "left", prev_tip_point_L, blur_L, kalman_filter_L)
                locationofdart_C, prev_tip_point_C = getRealLocation(corners_final_C, "center", prev_tip_point_C, blur_C, kalman_filter_C)

                if isinstance(locationofdart_R, tuple) and len(locationofdart_R) == 2:
                    cv2.circle(blur_R, locationofdart_R, 10, (255, 255, 255), 2, 8)
                    cv2.circle(blur_R, locationofdart_R, 2, (0, 255, 0), 2, 8)
                    print(f"Right Camera - Dart Location: {locationofdart_R}")

                if isinstance(locationofdart_L, tuple) and len(locationofdart_L) == 2:
                    cv2.circle(blur_L, locationofdart_L, 10, (255, 255, 255), 2, 8)
                    cv2.circle(blur_L, locationofdart_L, 2, (0, 255, 0), 2, 8)
                    print(f"Left Camera - Dart Location: {locationofdart_L}")

                if isinstance(locationofdart_C, tuple) and len(locationofdart_C) == 2:
                    cv2.circle(blur_C, locationofdart_C, 10, (255, 255, 255), 2, 8)
                    cv2.circle(blur_C, locationofdart_C, 2, (0, 255, 0), 2, 8)
                    print(f"Center Camera - Dart Location: {locationofdart_C}")

            except Exception as e:
                print(f"Something went wrong in finding the dart's location: {str(e)}")
                continue

            cv2.imshow("Dart Detection - Right", blur_R)
            cv2.imshow("Dart Detection - Left", blur_L)
            cv2.imshow("Dart Detection - Center", blur_C)

            # Update the reference frames after a dart has been detected
            success, t_R = cam2gray(cam_R)
            _, t_L = cam2gray(cam_L)
            _, t_C = cam2gray(cam_C)

        else:
            if cv2.countNonZero(thresh_R) > takeout_threshold or cv2.countNonZero(thresh_L) > takeout_threshold or cv2.countNonZero(thresh_C) > takeout_threshold:
                print("Takeout procedure initiated.")
                # Perform takeout actions here, such as resetting variables or updating the reference frames
                prev_tip_point_R = None
                prev_tip_point_L = None
                prev_tip_point_C = None

                # Wait for the specified delay to allow hand removal
                start_time = time.time()
                while time.time() - start_time < takeout_delay:
                    success, t_R = cam2gray(cam_R)
                    _, t_L = cam2gray(cam_L)
                    _, t_C = cam2gray(cam_C)
                    time.sleep(0.1)

                print("Takeout procedure completed.")

        if cv2.waitKey(10) == 27:
            break

    cam_R.release()
    cam_L.release()
    cam_C.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main() 
    main()
