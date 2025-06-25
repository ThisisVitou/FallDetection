## import library
import mediapipe as mp
import random
import time
import cv2
import os
import numpy as np

data_path = os.path.join('30_data')
actions = np.array(['falling0', 'lying', 'sitting', 'standing'])

no_videos = 400
video_length = 30

main_folder = 'raw_data2' ## folder that store raw data in this case video
destin_folder = '30_data' ## where the npy file will save.

# Import MediaPipe's pose estimation and drawing utilities
mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mpPose.Pose() # Create a pose estimator object (with default settings)

# Loop through each action label
for action in actions:
    print(f"Processing action: {action}")

    # Construct path to videos and destination folder for this action
    src_folder = os.path.join(main_folder, action)
    save_action_folder = os.path.join(destin_folder, action)
    os.makedirs(save_action_folder, exist_ok=True) # Create folder if it doesn't exist

    # Counters for tracking how many videos are processed/saved
    collected = 0
    video_id = 0
    saved_video_idx = 0

    # Loop until desired number of videos is collected
    while collected < no_videos:
        file_location = os.path.join(src_folder, f'video_{video_id}.avi')

        # If video file is missing, skip it
        if not os.path.exists(file_location):
            video_id += 1
            if video_id > 10000: # Stop search after too many attempts
                print(f"Too many missing videos, stopping search for {action}")
                break
            continue

        # Open the video file
        cap = cv2.VideoCapture(file_location)
        frame_landmarks = [] # Store pose landmarks from each frame
        print(f"Processing {action} video {video_id} as index {saved_video_idx}...")

        # Read frames one by one
        while True:
            success, img = cap.read()
            if not success:
                break # Stop if end of video

            # Convert BGR to RGB for MediaPipe
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB) # Get pose landmarks

            # If pose landmarks are found, extract them
            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS) # (optional) draw pose (only if you show the video)
                pose_tmp = [np.array([res.x, res.y, res.z, res.visibility]) for res in results.pose_landmarks.landmark]
            else:
                pose_tmp = [np.zeros(4) for _ in range(33)] # make it 0 if pose not detected

            # Save the frame's flattened landmark array
            frame_landmarks.append(np.array(pose_tmp).flatten())

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release() # Close the video file
        
        frame_count = len(frame_landmarks)

        # Skip short videos with less than 25 frames
        if frame_count < 25:
            print(f"Video {video_id} skipped, too short ({frame_count} frames).")
            video_id += 1
            continue

        # Normalize all videos to exactly 30 frames
        if 25 <= frame_count < 30:
            # Pad with last frame
            padding_needed = 30 - frame_count
            last_frame = frame_landmarks[-1]
            selected = frame_landmarks + [last_frame] * padding_needed
        elif 30 <= frame_count < 35:
            # Trim to 30 frames
            selected = frame_landmarks[:30]
        elif frame_count >= 35:
            # Trim to 30 frames
            selected = frame_landmarks[:30]
        else:
            selected = frame_landmarks  # Exact 30 frames

        # Save the processed frames as .npy files, one per frame
        save_to = os.path.join(save_action_folder, str(saved_video_idx))
        os.makedirs(save_to, exist_ok=True)

        for i, frame in enumerate(selected):
            np.save(os.path.join(save_to, f'{i}.npy'), frame)

        # Update counters
        collected += 1
        saved_video_idx += 1
        video_id += 1

    print(f"Finished {action}, total saved: {collected}")

cv2.destroyAllWindows() # Clean up any OpenCV windows (if used)
