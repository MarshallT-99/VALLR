import whisper
import pronouncing  # For converting words to phonemes

# Load Whisper model (you can use 'base', 'small', 'medium', 'large' depending on your needs)
model = whisper.load_model("base")

# Load and transcribe the audio using Whisper
def transcribe_audio(audio_file):
    result = model.transcribe(audio_file, word_timestamps=True)
    return result['segments']

# Convert words to phonemes using pronouncing library
def word_to_phonemes(word):
    phonemes_list = pronouncing.phones_for_word(word)
    if phonemes_list:
        return phonemes_list[0].split()  # Return the first pronunciation, split into phonemes
    return []

# Align phonemes with word timestamps
def align_phonemes_to_timestamps(segments):
    aligned_phonemes = []
    for segment in segments:
        words = segment['text'].split()
        for word_data in segment['words']:
            word = word_data['word'].strip().lower()
            start_time = word_data['start']
            end_time = word_data['end']
            phonemes = word_to_phonemes(word)
            
            if phonemes:
                word_duration = end_time - start_time
                num_phonemes = len(phonemes)
                phoneme_duration = word_duration / num_phonemes
                
                # Assign timestamps to each phoneme
                for i, phoneme in enumerate(phonemes):
                    phoneme_start = start_time + i * phoneme_duration
                    phoneme_end = phoneme_start + phoneme_duration
                    aligned_phonemes.append({
                        'phoneme': phoneme,
                        'start': phoneme_start,
                        'end': phoneme_end,
                        'word': word
                    })
    
    return aligned_phonemes

# Example usage
audio_file = "path_to_audio_file.mp3"
segments = transcribe_audio(audio_file)
phoneme_timestamps = align_phonemes_to_timestamps(segments)

# Print phoneme-level alignments
for p in phoneme_timestamps:
    print(f"Phoneme: {p['phoneme']}, Start: {p['start']}, End: {p['end']}, Word: {p['word']}")
