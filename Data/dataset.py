import os
import torch
import numpy as np
from torch.utils.data import Dataset
from decord import VideoReader, cpu
import pronouncing  # Import the pronouncing library
import re

class VideoDataset(Dataset):
    def __init__(self, video_dir, phoneme_vocab, split="train", num_frames=16, frame_size=(224, 224)):
        self.video_dir = video_dir
        self.split = split
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.phoneme_vocab = phoneme_vocab  # Phoneme to index mapping

        self.video_paths = []
        self.labels = []

        for class_name in os.listdir(video_dir):
            class_path = os.path.join(video_dir, class_name)
            count = 0
            if os.path.isdir(class_path):
                split_path = os.path.join(class_path, split)
                if os.path.isdir(split_path):
                    for video_name in os.listdir(split_path):
                        # if count == 10:
                        #     break
                        # count += 1
                        video_path = os.path.join(split_path, video_name)
                        if video_name.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                            self.video_paths.append(video_path)
                            self.labels.append(class_name)

        if len(self.video_paths) == 0:
            raise ValueError(f"No video files found in directory: {video_dir} for split: {split}")

        print(f"Loaded {len(self.video_paths)} videos from {split} split in {video_dir}.")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        # Retry logic if the sample is invalid
        max_retries = 3
        retries = 0

        while retries < max_retries:
            video_path = self.video_paths[idx]
            label = self.labels[idx]

            video = load_and_preprocess_video(video_path, self.num_frames)

            # If the video is None, try the next one (increment index)
            if video is None:
                retries += 1
                idx = (idx + 1) % len(self.video_paths)  # Increment index to the next item
                continue

            video_tensor = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to (T, C, H, W)

            # Get phoneme indices for the label using the pronouncing library
            phonemes = get_phonemes(label)
            # print(phonemes)
            phoneme_indices = [self.phoneme_vocab.get(phoneme) for phoneme in phonemes if phoneme in self.phoneme_vocab]

            # If any phoneme was unknown or invalid, try the next item
            if len(phoneme_indices) == 0:
                retries += 1
                idx = (idx + 1) % len(self.video_paths)
                continue

            phoneme_tensor = torch.tensor(phoneme_indices)

            # print(phoneme_tensor)
            # print(label)

            return video_tensor, phoneme_tensor

        # If max retries exceeded, raise an error
        raise ValueError(f"Exceeded max retries for video at index {idx}")

def load_and_preprocess_video(video_path, num_frames):
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=4)
    except Exception as e:
        print(f"Error loading video: {video_path}. Error: {e}")
        return None

    frame_count = len(vr)
    sample_indices = np.linspace(0, frame_count - 1, num_frames).astype(int)

    frames = []
    for idx in sample_indices:
        frame = vr[idx].asnumpy()
        frames.append(frame)

    if len(frames) < num_frames:
        return None

    video_np = np.array(frames)  # (T, H, W, C)
    return video_np

def get_phonemes(word):
    """
    Use the 'pronouncing' library to convert the word into phonemes, removing any numbers from phonemes.

    Args:
        word (str): Input word to convert to phonemes.

    Returns:
        list: List of phonemes corresponding to the input word with numbers removed.
    """
    word = word.lower()
    phones = pronouncing.phones_for_word(word)  # Get phonemes for the word
    if phones:
        # Remove any numbers from phonemes using regex
        return [re.sub(r'\d+', '', phoneme) for phoneme in phones[0].split()]  # Return phonemes for the first pronunciation
    else:
        return []  # Return empty list if no phoneme is found