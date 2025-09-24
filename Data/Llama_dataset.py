import pronouncing
import random
import csv
import os
import re
from tqdm import tqdm

def read_file_line_by_line(file_path):
    """
    Reads a text file line by line and prints each line.

    Args:
        file_path (str): Path to the text file.

    Returns:
        list: A list of strings, where each string is a line from the file.
    """
    lines = []
    
    # Open the file in read mode
    with open(file_path, 'r') as file:
        for line in file:
            lines.append(line.strip())  # Add each line to the list, stripping the newline characters
    
    return lines

# def get_phonemes(word, mask_prob=0.2, mask_token="<mask>"):
#     """
#     Use the 'pronouncing' library to convert the word into phonemes, with optional masking.

#     Args:
#         word (str): Input word to convert to phonemes.
#         mask_prob (float): Probability of masking a phoneme. Default is 0.2 (20% chance).
#         mask_token (str): Token used for masking. Default is '<mask>'.

#     Returns:
#         str: String of phonemes with some masked based on the mask_prob.
#     """
#     word = word.lower()
#     phones = pronouncing.phones_for_word(word)  # Get phonemes for the word
    
#     if phones:
#         phonemes = phones[0].split()  # Split the phonemes into a list

#         # Apply masking with a probability of 'mask_prob'
#         masked_phonemes = [
#             phoneme if random.random() > mask_prob else mask_token 
#             for phoneme in phonemes
#         ]

#         return " ".join(masked_phonemes)  # Join the masked phonemes into a single string
#     else:
#         return ""  # Return an empty string if no phoneme is found

def get_phonemes(sentence):
    """
    Use the 'pronouncing' library to convert a sentence into phonemes, removing any numbers from phonemes.

    Args:
        sentence (str): Input sentence to convert to phonemes.

    Returns:
        list: List of phonemes corresponding to the input sentence with numbers removed.
    """
    sentence = sentence.lower()
    phoneme_list = []
    words = sentence.split()  # Split the sentence into words

    for word in words:
        phones = pronouncing.phones_for_word(word)  # Get phonemes for the word
        if phones:
            # Remove any numbers from phonemes using regex and add to the phoneme list
            phoneme_list.extend([re.sub(r'\d+', '', phoneme) for phoneme in phones[0].split()])

    return phoneme_list
    
def extract_text_from_file(file_path):
    """
    Extracts the text after 'Text:' from the given file.
    Args:
        file_path (str): Path to the text file.
    Returns:
        str: The extracted text.
    """
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("Text:"):
                return line.split("Text:")[1].strip()
    return None
    
def process_directory(directory_path, csv_file_path):
    """
    Processes all .txt files in subdirectories of the given directory.
    Args:
        directory_path (str): Path to the directory.
    """
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Sentence", "Phonemes"])  # Write CSV header
        for root, dirs, files in tqdm(os.walk(directory_path), desc="Extracting Phonemes", leave=False):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    text = extract_text_from_file(file_path)  # Extract text from file
                    if text:
                        phonemes = get_phonemes(text)  # Extract phonemes from the text
                        phonemes_str = ' '.join(phonemes)  # Convert phonemes list to a space-separated string
                        writer.writerow([text, phonemes_str])  # Write to CSV



def main():
    file_path = "../../Data/lrs3/lrs3_trainval/trainval"  # Input text file with words
    csv_file_path = "phonemes.csv"  # Output CSV file to store word-phoneme mappings

    process_directory(file_path, csv_file_path)  # Process all .txt files in the directory

    # Read words from the input file
    # lines = read_file_line_by_line(file_path)

    # Open the CSV file once and append the data for each word
    # with open(csv_file_path, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Word", "Phonemes"])  # Write CSV header

    #     for word in lines:
    #         phonemes = get_phonemes(word)  # Get phonemes with optional masking
    #         writer.writerow([word, phonemes])  # Write word and corresponding phonemes to CSV

    print(f"Phonemes for words written to {csv_file_path}")

if __name__ == "__main__":
    main()