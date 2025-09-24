import os
import cv2
from decord import VideoReader, cpu
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('..')
from face_cropper import FaceCropper
from config import load_args


class Preprocess:
    def __init__(self, video_dir, output_dir, split="train", frame_size=(224, 224), output_format='mp4'):
        self.video_path = video_dir
        self.output_dir = output_dir
        self.split = split
        self.frame_size = frame_size
        self.output_format = output_format
        self.class_to_index = {}
        self.video_paths = []
        self.labels = []
        # Load the FaceCropper
        self.face_cropper = FaceCropper(min_face_detector_confidence=0.5, face_detector_model_selection="SHORT_RANGE", 
                                        landmark_detector_static_image_mode="STATIC_MODE", min_landmark_detector_confidence=0.5)

    def get_video_path(self):
        for class_name in tqdm(os.listdir(self.video_path), desc="Video Loading", leave=False):
            class_path = os.path.join(self.video_path, class_name)
            count = 0
            if os.path.isdir(class_path):  # Check if it's a directory
                if class_name not in self.class_to_index:
                    self.class_to_index[class_name] = len(self.class_to_index)
                split_path = os.path.join(class_path, self.split)
                if os.path.isdir(split_path):
                    for video_name in os.listdir(split_path):
                        count += 1
                        video_path = os.path.join(split_path, video_name)
                        # if count > 10:
                        #     break
                        if video_name.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Valid video formats
                            self.video_paths.append((video_path, class_name, video_name))
                            self.labels.append(self.class_to_index[class_name])
                        elif video_name.endswith('.txt'):  # Check for label files
                            continue
                        else:
                            print(f"Skipped file with unsupported format: {video_name}")

    def create_output_dir(self, class_name):
        # Create a corresponding directory structure in the output folder
        class_output_dir = os.path.join(self.output_dir, class_name, self.split)
        if not os.path.exists(class_output_dir):
            os.makedirs(class_output_dir)
        return class_output_dir

    def save_preprocessed_video(self, frames, output_path, fps=30):
        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Save as .mp4 file
        height, width, _ = frames[0].shape
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            video_writer.write(frame)  # Write each frame to the video file

        video_writer.release()  # Release the writer
        # print(f"Saved preprocessed video to {output_path}")

    def load_and_preprocess_video(self, video_path):
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        except Exception as e:
            print(f"Error loading video: {video_path}. Error: {e}")
            return None

        frames = []
        for frame in vr:  # Iterate over all frames in the video
            frame = frame.asnumpy()
            frame = cv2.resize(frame, self.frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            faces = self.face_cropper.get_faces(frame, remove_background=False, correct_roll=True)
            if faces:  # If faces are detected
                frame = faces[0]  # Use the first detected face
                frame = cv2.resize(frame, self.frame_size)
            else:
                frame = cv2.resize(frame, self.frame_size)

            frames.append(frame)

        if len(frames) == 0:
            return None  # Skip video if no frames are found

        return frames

    def process_videos(self):
        self.get_video_path()
        for video_path, class_name, video_name in tqdm(self.video_paths, desc="Video Preprocessing", leave=False):
            video_frames = self.load_and_preprocess_video(video_path)
            if video_frames is not None:
                # Create output directory structure
                class_output_dir = self.create_output_dir(class_name)

                # Save the video in the specified format (.mp4 by default)
                output_path = os.path.join(class_output_dir, os.path.splitext(video_name)[0] + f".{self.output_format}")
                self.save_preprocessed_video(video_frames, output_path)
            else:
                print(f"Skipping video: {video_name}")

def main():
    # Extract args
    args = load_args()

    # Extract specific values from args
    video_dir = args.videos_root
    output_dir = args.videos_output
    preprocessor = Preprocess(video_dir=video_dir, output_dir=output_dir, split="val")
    preprocessor.process_videos()

if __name__ == '__main__':
    main()