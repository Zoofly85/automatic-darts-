import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import math
import time
from shapely.geometry import Point, LineString, Polygon
from typing import List
from game_501 import update_score, check_game_over

# Constants
NUM_CAMERAS = 3
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
DARTBOARD_DIAMETER_MM = 451
DOUBLE_RING_OUTER_RADIUS_MM = 170

# Dartboard radii in mm
BULLSEYE_RADIUS_MM = 6.35
OUTER_BULL_RADIUS_MM = 15.9
TRIPLE_RING_INNER_RADIUS_MM = 99
TRIPLE_RING_OUTER_RADIUS_MM = 107
DOUBLE_RING_INNER_RADIUS_MM = 162
DOUBLE_RING_OUTER_RADIUS_MM = 170

# Dart tip radius in mm
TIP_RADIUS_MM = 1.15

# Kalman filter parameters
DT = 1.0 / 30.0  # Assuming 30 FPS
U_X = 0
U_Y = 0
STD_ACC = 1.0
X_STD_MEAS = 0.1
Y_STD_MEAS = 0.1

# Takeout parameters
TAKEOUT_THRESHOLD = 18000
TAKEOUT_DELAY = 3.0

# Global variables
dartboard_image = None
score_images = None
perspective_matrices = []
center = (IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2)

# Helper functions

def draw_segment_text(image, center, start_angle, end_angle, radius, text):
    angle = (start_angle + end_angle) / 2
    text_x = int(center[0] + radius * np.cos(angle))
    text_y = int(center[1] + radius * np.sin(angle))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_origin = (text_x - text_size[0] // 2, text_y + text_size[1] // 2)
    cv2.putText(image, text, text_origin, font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

def draw_point_at_angle(image, center, angle_degrees, radius, color, point_radius):
    angle_radians = np.radians(angle_degrees)
    x = int(center[0] + radius * np.cos(angle_radians))
    y = int(center[1] - radius * np.sin(angle_radians))
    cv2.circle(image, (x, y), point_radius, color, -1)

def calculate_score(distance, angle):
    if angle < 0:
        angle += 2 * np.pi
    sector_scores = [10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5, 20, 1, 18, 4, 13, 6]
    sector_index = int(angle / (2 * np.pi) * 20)
    base_score = sector_scores[sector_index]
    if distance <= BULLSEYE_RADIUS_PX:
        return 50
    elif distance <= OUTER_BULL_RADIUS_PX:
        return 25
    elif TRIPLE_RING_INNER_RADIUS_PX < distance <= TRIPLE_RING_OUTER_RADIUS_PX:
        return base_score * 3
    elif DOUBLE_RING_INNER_RADIUS_PX < distance <= DOUBLE_RING_OUTER_RADIUS_PX:
        return base_score * 2
    elif distance <= DOUBLE_RING_OUTER_RADIUS_PX:
        return base_score
    else:
        return 0

def select_points_event(event, x, y, flags, param):
    frame, selected_points, camera_index = param
    if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 4:
        selected_points.append([x, y])
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow(f"Camera {camera_index} - Select 4 Points", frame)
        if len(selected_points) == 4:
            cv2.destroyWindow(f"Camera {camera_index} - Select 4 Points")

def calibrate_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()
    if ret:
        window_name = f"Camera {camera_index} - Select 4 Points"
        cv2.namedWindow(window_name)
        cv2.imshow(window_name, frame)
        
        selected_points = []
        cv2.setMouseCallback(window_name, select_points_event, (frame, selected_points, camera_index))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cap.release()

        if len(selected_points) == 4:
            return np.float32(selected_points)
    return None

def calculate_score_from_coordinates(x, y, camera_index):
    inverse_matrix = cv2.invert(perspective_matrices[camera_index])[1]
    transformed_coords = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), inverse_matrix)[0][0]
    transformed_x, transformed_y = transformed_coords

    dx = transformed_x - center[0]
    dy = transformed_y - center[1]
    distance_from_center = math.sqrt(dx**2 + dy**2)
    angle = math.atan2(dy, dx)
    score = calculate_score(distance_from_center, angle)
    return score

def click_event(event, x, y, flags, param):
    gui = param  # Get the gui object passed as a parameter

    if event == cv2.EVENT_LBUTTONDOWN:
        if gui.selected_score_index is not None:
            # Calculate the score based on the clicked coordinates
            dx = x - center[0]
            dy = y - center[1]
            distance_from_center = math.sqrt(dx**2 + dy**2)
            angle = math.atan2(dy, dx)
            score = calculate_score(distance_from_center, angle)

            # Update the score in the dart_scores list
            old_score = gui.dart_scores[gui.selected_score_index]
            gui.dart_scores[gui.selected_score_index] = score

            # Update the current score based on the score correction
            if old_score is not None:
                gui.current_score += old_score  # Add back the old score
            gui.current_score = update_score(gui.current_score, [score])  # Subtract the new score

            gui.increment_score_fixed_count()  # Increment the score fixed count

            gui.selected_score_index = None
        else:
            # Check if a score box was clicked
            for i, score in enumerate(gui.dart_scores):
                text_x = 50
                text_y = 100 + i * 50
                text_size = cv2.getTextSize(f"Dart {i+1}: {score}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                if text_x - 10 < x < text_x + text_size[0] + 10 and text_y - text_size[1] - 10 < y < text_y + 10:
                    gui.correct_score(i)
                    break
            
def load_perspective_matrices():
    perspective_matrices = []
    for camera_index in range(NUM_CAMERAS):
        try:
            data = np.load(f'perspective_matrix_camera_{camera_index}.npz')
            matrix = data['matrix']
            perspective_matrices.append(matrix)
        except FileNotFoundError:
            print(f"Perspective matrix file not found for camera {camera_index}. Please calibrate the cameras first.")
            exit(1)
    return perspective_matrices

def cam2gray(cam):
    success, image = cam.read()
    img_g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return success, img_g

def getThreshold(cam, t):
    success, t_plus = cam2gray(cam)
    dimg = cv2.absdiff(t, t_plus)
    blur = cv2.GaussianBlur(dimg, (5, 5), 0)
    blur = cv2.bilateralFilter(blur, 9, 75, 75)
    _, thresh = cv2.threshold(blur, 60, 255, 0)
    return thresh

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

        # Adjust the tip coordinates by half of the tip's diameter
        tip_radius_px = TIP_RADIUS_MM * PIXELS_PER_MM

        # Determine the adjustment direction based on the camera's perspective
        adjustment_direction = 0  # Adjust towards the dartboard center (negative direction)

        # Calculate the adjusted tip coordinates
        adjusted_tip_x = lowest_point[0] + adjustment_direction * tip_radius_px
        adjusted_tip_y = lowest_point[1]

        # Predict the dart tip position using the Kalman filter
        predicted_tip = kalman_filter.predict()
        
        # Update the Kalman filter with the observed dart tip position
        kalman_filter.update(np.array([[adjusted_tip_x], [adjusted_tip_y]]))
        
        return int(adjusted_tip_x), int(adjusted_tip_y)
    
    return None



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
            cv2.circle(blur, (tip_x, tip_y), 5, (0, 255, 0), 2)
        
        locationofdart = dart_tip
    
    return locationofdart, dart_tip

class DartboardGUI:
    def __init__(self, master):
        self.master = master
        master.title("Dartboard Score Detection")

        self.calibrate_button = tk.Button(master, text="Calibrate", command=self.calibrate)
        self.calibrate_button.pack(pady=10)

        self.game_button = tk.Button(master, text="Start Game", command=self.start_game)
        self.game_button.pack(pady=10)

        self.game_501_button = tk.Button(master, text="501", command=self.start_game_501)
        self.game_501_button.pack(pady=10)

        self.reset_button = tk.Button(master, text="Reset Counters", command=self.reset_counters)
        self.reset_button.pack(pady=10)

        self.dart_scores = [None] * 3  # Initialize dart score boxes
        self.selected_score_index = None  # Index of the selected score for correction
        self.current_score = 501  # Current score for the 501 game
        self.bust = False  # Flag to indicate if a bust has occurred
        self.last_valid_score = 501  # Last valid score before a bust occurred
        self.total_darts_thrown = 0  # Total number of darts thrown
        self.total_scores_fixed = 0  # Total number of scores fixed

        self.load_counters()  # Load the counter values from file when the GUI is initialized

    def calibrate(self):
        messagebox.showinfo("Calibration", "Please select 4 points on each camera feed.")
        
        # Define the drawn_points variable
        global drawn_points
        drawn_points = np.float32([
            [center[0], center[1] - DOUBLE_RING_OUTER_RADIUS_PX],
            [center[0] + DOUBLE_RING_OUTER_RADIUS_PX, center[1]],
            [center[0], center[1] + DOUBLE_RING_OUTER_RADIUS_PX],
            [center[0] - DOUBLE_RING_OUTER_RADIUS_PX, center[1]],
        ])
        
        for camera_index in range(NUM_CAMERAS):
            live_feed_points = calibrate_camera(camera_index)
            if live_feed_points is not None:
                M = cv2.getPerspectiveTransform(drawn_points, live_feed_points)
                perspective_matrices.append(M)
                np.savez(f'perspective_matrix_camera_{camera_index}.npz', matrix=M)
            else:
                messagebox.showerror("Calibration Error", f"Failed to calibrate camera {camera_index}")
                return

        messagebox.showinfo("Calibration", "Calibration completed successfully.")

    def start_game(self):
        messagebox.showinfo("Game", "Starting the dartboard game.")
        main(mode="standard")

    def start_game_501(self):
        messagebox.showinfo("501 Game", "Starting the 501 game.")
        self.current_score = 501  # Reset the current score
        self.bust = False  # Reset the bust flag
        self.last_valid_score = 501  # Reset the last valid score
        main(mode="501")

    def correct_score(self, index):
        self.selected_score_index = index

    def reset_counters(self):
        self.total_darts_thrown = 0
        self.total_scores_fixed = 0
        self.save_counters()  # Save the reset counter values to file
        messagebox.showinfo("Reset Counters", "Dart count and score fixed count have been reset.")

    def increment_dart_count(self):
        self.total_darts_thrown += 1
        self.save_counters()  # Save the updated counter values to file

    def increment_score_fixed_count(self):
        self.total_scores_fixed += 1
        self.save_counters()  # Save the updated counter values to file

    def calculate_accuracy(self):
        if self.total_darts_thrown > 0:
            return (1 - (self.total_scores_fixed / self.total_darts_thrown)) * 100
        else:
            return 100.0

    def save_counters(self):
        with open("counter_data.txt", "w") as file:
            file.write(f"{self.total_darts_thrown},{self.total_scores_fixed}")

    def load_counters(self):
        try:
            with open("counter_data.txt", "r") as file:
                data = file.read().strip().split(",")
                if len(data) == 2:
                    self.total_darts_thrown = int(data[0])
                    self.total_scores_fixed = int(data[1])
        except FileNotFoundError:
            pass  # If the file doesn't exist, use the default counter values
        
        
def main(mode="standard"):
    global dartboard_image, score_images, perspective_matrices

    perspective_matrices = load_perspective_matrices()

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
    kalman_filter_R = KalmanFilter(DT, U_X, U_Y, STD_ACC, X_STD_MEAS, Y_STD_MEAS)
    kalman_filter_L = KalmanFilter(DT, U_X, U_Y, STD_ACC, X_STD_MEAS, Y_STD_MEAS)
    kalman_filter_C = KalmanFilter(DT, U_X, U_Y, STD_ACC, X_STD_MEAS, Y_STD_MEAS)

    camera_scores = [None] * NUM_CAMERAS  # Initialize camera_scores list
    majority_score = None
    dart_coordinates = None

    dart_count = 0  # Counter for the number of darts thrown

    gui.load_counters()  # Load the counter values when the program starts

    while success:
        for camera_index, cam in enumerate([cam_R, cam_L, cam_C]):
            ret, frame = cam.read()
            if not ret:
                break

        time.sleep(0.1)
        thresh_R = getThreshold(cam_R, t_R)
        thresh_L = getThreshold(cam_L, t_L)
        thresh_C = getThreshold(cam_C, t_C)

        if (cv2.countNonZero(thresh_R) > 1000 and cv2.countNonZero(thresh_R) < 7500) or \
                (cv2.countNonZero(thresh_L) > 1000 and cv2.countNonZero(thresh_L) < 7500) or \
                (cv2.countNonZero(thresh_C) > 1000 and cv2.countNonZero(thresh_C) < 7500):
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

                for camera_index, locationofdart in enumerate([locationofdart_R, locationofdart_L, locationofdart_C]):
                    if isinstance(locationofdart, tuple) and len(locationofdart) == 2:
                        x, y = locationofdart
                        score = calculate_score_from_coordinates(x, y, camera_index)
                        print(f"Camera {camera_index} - Dart Location: {locationofdart}, Score: {score}")

                        # Store the score in the camera_scores list
                        camera_scores[camera_index] = score

                # Apply majority rule to determine the final score
                final_score = None
                score_counts = {}
                for score in camera_scores:
                    if score is not None:
                        if score in score_counts:
                            score_counts[score] += 1
                        else:
                            score_counts[score] = 1

                if score_counts:
                    final_score = max(score_counts, key=score_counts.get)
                    majority_score = final_score

                    # Find the camera with the majority score
                    majority_camera_index = camera_scores.index(final_score)
                    dart_coordinates = (locationofdart_R, locationofdart_L, locationofdart_C)[majority_camera_index]

                    # Transform the dart coordinates to match the drawn dartboard
                    if dart_coordinates is not None:
                        x, y = dart_coordinates
                        inverse_matrix = cv2.invert(perspective_matrices[majority_camera_index])[1]
                        transformed_coords = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), inverse_matrix)[0][0]
                        dart_coordinates = tuple(map(int, transformed_coords))

                    if mode == "501":
                        # Update the dart scores and current score only if a bust has not occurred
                        if not gui.bust:
                            # Update the dart scores
                            if dart_count < 3:
                                gui.dart_scores[dart_count] = majority_score
                                dart_count += 1
                                gui.increment_dart_count()  # Increment the total dart count

                            # Update the current score based on the score correction
                            if gui.selected_score_index is None:
                                new_score = update_score(gui.current_score, [majority_score])
                                if new_score < 0:
                                    print("Bust! Score cannot go below zero.")
                                    gui.bust = True
                                else:
                                    gui.current_score = new_score
                                    gui.last_valid_score = gui.current_score
                                    
                                    # Check if the game is over
                        if check_game_over(gui.current_score):
                            print("Game Over! You won!")
                            break

                if final_score is not None:
                    print(f"Final Score (Majority Rule): {final_score}")
                else:
                    print("No majority score found.")

            except Exception as e:
                print(f"Something went wrong in finding the dart's location: {str(e)}")
                continue

            # Update the reference frames after a dart has been detected
            success, t_R = cam2gray(cam_R)
            _, t_L = cam2gray(cam_L)
            _, t_C = cam2gray(cam_C)

        else:
            if cv2.countNonZero(thresh_R) > TAKEOUT_THRESHOLD or cv2.countNonZero(thresh_L) > TAKEOUT_THRESHOLD or cv2.countNonZero(thresh_C) > TAKEOUT_THRESHOLD:
                print("Takeout procedure initiated.")
                # Perform takeout actions here, such as resetting variables or updating the reference frames
                prev_tip_point_R = None
                prev_tip_point_L = None
                prev_tip_point_C = None
                majority_score = None
                dart_coordinates = None

                if mode == "501":
                    # Reset the dart scores and count for the 501 game mode
                    gui.dart_scores = [None] * 3
                    dart_count = 0

                    # Restore the last valid score if a bust occurred
                    if gui.bust:
                        gui.current_score = gui.last_valid_score
                        gui.bust = False

                # Wait for the specified delay to allow hand removal
                start_time = time.time()
                while time.time() - start_time < TAKEOUT_DELAY:
                    success, t_R = cam2gray(cam_R)
                    _, t_L = cam2gray(cam_L)
                    _, t_C = cam2gray(cam_C)
                    time.sleep(0.1)

                print("Takeout procedure completed.")

        if mode == "501":
            # Display the dart scores above the dartboard image
            dartboard_image_copy = dartboard_image.copy()
            for i, score in enumerate(gui.dart_scores):
                text_x = 50
                text_y = 100 + i * 50
                text_size = cv2.getTextSize(f"Dart {i+1}: {score}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                box_color = (255, 0, 0) if i == gui.selected_score_index else (255, 255, 0)  # Blue color for the box, yellow if selected
                cv2.rectangle(dartboard_image_copy, (text_x - 10, text_y - text_size[1] - 10),
                              (text_x + text_size[0] + 10, text_y + 10), box_color, -1)
                cv2.putText(dartboard_image_copy, f"Dart {i+1}: {score}", (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)  # Black color for the text

            # Display the current score or "BUST" in the score box
            if gui.bust:
                score_text = "BUST"
                score_box_color = (0, 0, 255)  # Red color for bust
            else:
                score_text = str(gui.current_score)
                score_box_color = (0, 255, 255)  # Yellow color for valid score

            score_text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            score_text_x = 50
            score_text_y = 400
            cv2.rectangle(dartboard_image_copy, (score_text_x - 10, score_text_y - score_text_size[1] - 10),
                          (score_text_x + score_text_size[0] + 10, score_text_y + 10), score_box_color, -1)
            cv2.putText(dartboard_image_copy, score_text, (score_text_x, score_text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)  # Black color for the text

            # Display the dart count, score fixed count, and accuracy percentage
            dart_count_text = f"Darts Thrown: {gui.total_darts_thrown}"
            score_fixed_count_text = f"Scores Fixed: {gui.total_scores_fixed}"
            accuracy_text = f"Accuracy: {gui.calculate_accuracy():.2f}%"
            dart_count_text_size = cv2.getTextSize(dart_count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            score_fixed_count_text_size = cv2.getTextSize(score_fixed_count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            accuracy_text_size = cv2.getTextSize(accuracy_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            dart_count_text_x = 50
            dart_count_text_y = 450
            score_fixed_count_text_x = 50
            score_fixed_count_text_y = 475
            accuracy_text_x = 50
            accuracy_text_y = 500
            cv2.putText(dartboard_image_copy, dart_count_text, (dart_count_text_x, dart_count_text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(dartboard_image_copy, score_fixed_count_text, (score_fixed_count_text_x, score_fixed_count_text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(dartboard_image_copy, accuracy_text, (accuracy_text_x, accuracy_text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

            if majority_score is not None:
                cv2.putText(dartboard_image_copy, f"Majority Score: {majority_score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            if dart_coordinates is not None:
                x, y = dart_coordinates
                cv2.circle(dartboard_image_copy, (int(x), int(y)), 5, (0, 0, 255), -1)

            cv2.imshow('Dartboard', dartboard_image_copy)
            cv2.setMouseCallback('Dartboard', click_event, param=gui)  # Pass the gui object to click_event
        else:
            # Display the scores and dart coordinates on the dartboard image
            dartboard_image_copy = dartboard_image.copy()
            if majority_score is not None:
                cv2.putText(dartboard_image_copy, f"Majority Score: {majority_score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if dart_coordinates is not None:
                x, y = dart_coordinates
                cv2.circle(dartboard_image_copy, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.imshow('Dartboard', dartboard_image_copy)
            cv2.setMouseCallback('Dartboard', click_event, param=gui)  # Pass the gui object to click_event

        key = cv2.waitKey(1) & 0xFF

        # Check for 'q' (quit)
        if key == ord('q'):
            break

    caps = []  # Define the variable "caps" as an empty list
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()


def draw_dartboard():
    global dartboard_image, PIXELS_PER_MM, BULLSEYE_RADIUS_PX, OUTER_BULL_RADIUS_PX, TRIPLE_RING_INNER_RADIUS_PX, TRIPLE_RING_OUTER_RADIUS_PX, DOUBLE_RING_INNER_RADIUS_PX, DOUBLE_RING_OUTER_RADIUS_PX

    # Calculate the conversion factor from mm to pixels
    PIXELS_PER_MM = IMAGE_HEIGHT / DARTBOARD_DIAMETER_MM

    # Convert mm to pixels
    BULLSEYE_RADIUS_PX = int(BULLSEYE_RADIUS_MM * PIXELS_PER_MM)
    OUTER_BULL_RADIUS_PX = int(OUTER_BULL_RADIUS_MM * PIXELS_PER_MM)
    TRIPLE_RING_INNER_RADIUS_PX = int(TRIPLE_RING_INNER_RADIUS_MM * PIXELS_PER_MM)
    TRIPLE_RING_OUTER_RADIUS_PX = int(TRIPLE_RING_OUTER_RADIUS_MM * PIXELS_PER_MM)
    DOUBLE_RING_INNER_RADIUS_PX = int(DOUBLE_RING_INNER_RADIUS_MM * PIXELS_PER_MM)
    DOUBLE_RING_OUTER_RADIUS_PX = int(DOUBLE_RING_OUTER_RADIUS_MM * PIXELS_PER_MM)

    # Create a blank image with white background
    dartboard_image = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8) * 255

    # Draw the bullseye and rings
    cv2.circle(dartboard_image, center, BULLSEYE_RADIUS_PX, (0, 0, 0), -1, lineType=cv2.LINE_AA)  # Bullseye
    cv2.circle(dartboard_image, center, OUTER_BULL_RADIUS_PX, (255, 0, 0), 2, lineType=cv2.LINE_AA)  # Outer bull
    cv2.circle(dartboard_image, center, TRIPLE_RING_INNER_RADIUS_PX, (0, 255, 0), 2, lineType=cv2.LINE_AA)  # Inner triple
    cv2.circle(dartboard_image, center, TRIPLE_RING_OUTER_RADIUS_PX, (0, 255, 0), 2, lineType=cv2.LINE_AA)  # Outer triple
    cv2.circle(dartboard_image, center, DOUBLE_RING_INNER_RADIUS_PX, (0, 0, 255), 2, lineType=cv2.LINE_AA)  # Inner double
    cv2.circle(dartboard_image, center, DOUBLE_RING_OUTER_RADIUS_PX, (0, 0, 255), 2, lineType=cv2.LINE_AA)  # Outer double

    # Draw the sector lines
    for angle in np.linspace(0, 2 * np.pi, 21)[:-1]:  # 20 sectors
        start_x = int(center[0] + np.cos(angle) * DOUBLE_RING_OUTER_RADIUS_PX)
        start_y = int(center[1] + np.sin(angle) * DOUBLE_RING_OUTER_RADIUS_PX)
        end_x = int(center[0] + np.cos(angle) * OUTER_BULL_RADIUS_PX)
        end_y = int(center[1] + np.sin(angle) * OUTER_BULL_RADIUS_PX)
        cv2.line(dartboard_image, (start_x, start_y), (end_x, end_y), (0, 0, 0), 1, lineType=cv2.LINE_AA)

    text_radius_px = int((TRIPLE_RING_OUTER_RADIUS_PX + DOUBLE_RING_INNER_RADIUS_PX) / 2)

    sector_scores = [10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5, 20, 1, 18, 4, 13, 6]
    for i, score in enumerate(sector_scores):
        start_angle = (i * 360 / 20 - 0) * np.pi / 180
        end_angle = ((i + 1) * 360 / 20 - 0) * np.pi / 180
        draw_segment_text(dartboard_image, center, start_angle, end_angle, text_radius_px, str(score))

    sector_intersections = {
        '20_1': 0,
        '6_10': 90,
        '19_3': 180,
        '11_14': 270,
    }

    for angle in sector_intersections.values():
        draw_point_at_angle(dartboard_image, center, angle, DOUBLE_RING_OUTER_RADIUS_PX, (255, 0, 0), 5)

if __name__ == "__main__":
    root = tk.Tk()
    gui = DartboardGUI(root)

    draw_dartboard()  # Draw the dartboard image

    # Create the score images
    score_images = [np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8) for _ in range(NUM_CAMERAS)]

    root.mainloop()
