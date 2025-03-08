import os
import time
import whisper
import streamlit as st
import torch
import ffmpeg
import openai

from utils.constants import REEL_DIR

# Function to convert video to audio
def video_to_audio(video_file, output_audio_file):
    try:
        progress_text = "Extracting audio..."
        my_bar = st.progress(10, text=progress_text)

        # Use FFmpeg to extract audio
        ffmpeg.input(video_file).output(output_audio_file).run(overwrite_output=True)
        
        my_bar.progress(100, text=progress_text)
        my_bar.empty()
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Function to convert audio to text using Whisper
def transcribe_audio(audio_file, output_text_file):
    progress_text = "Transcribing audio..."
    my_bar = st.progress(10, text=progress_text)

    # Check if a GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the Whisper model
    model = whisper.load_model("base").to(device)
    my_bar.progress(50, text=progress_text)

    # Transcribe the audio file
    result = model.transcribe(audio_file)
    my_bar.progress(80, text=progress_text)
    
    extracted_text = "".join(
        f"[{round(segment['start'], 2)} - {round(segment['end'], 2)}] {segment['text']}\n"
        for segment in result["segments"]
    )
    
    save_transcription_to_file(extracted_text, output_text_file)
    my_bar.progress(100, text=progress_text)
    my_bar.empty()
    
    return extracted_text

# Function to save the transcribed text into a file
def save_transcription_to_file(transcribed_text, output_file):
    with open(output_file, "w") as file:
        file.write(transcribed_text)
    print(f"Transcription saved to {output_file}")

# Function to cut video using FFmpeg
def cut_video(input_video_path, output_video_path, start_time, end_time, target_width, target_height):
    progress_text = "Generating reel..."
    my_bar = st.progress(10, text=progress_text)
    
    temp_reel_path = os.path.join(REEL_DIR, "temp_reel.mp4")
    
    # Cut the video using FFmpeg
    ffmpeg.input(input_video_path, ss=start_time, to=end_time).output(temp_reel_path).run(overwrite_output=True)
    my_bar.progress(50, text=progress_text)
    
    # Resize and pad the video to fit target dimensions
    ffmpeg.input(temp_reel_path).filter('scale', target_width, -1).filter('pad', target_width, target_height, '(ow-iw)/2', '(oh-ih)/2').output(output_video_path).run(overwrite_output=True)
    
    my_bar.progress(100, text=progress_text)
    my_bar.empty()

# Function to analyze text and extract key timestamps
def analyze_text(audio_file_path):
    progress_text = "Analyzing Video..."
    my_bar = st.progress(10, text=progress_text)
    
    openai.api_key = "YOUR_API_KEY"
    file_contents = ""
    if st.session_state.get("transcribed_text") is None:
        with open(audio_file_path, "r") as file:
            file_contents = file.read()
    else:
        file_contents = st.session_state["transcribed_text"]
    
    my_bar.progress(30, text=progress_text)
    
    analyzeSpeechPrompt = (
        file_contents
        + "\nYou are tasked with identifying the type of video based on its transcript. Provide a brief classification of the video type and rationale."
    )
    
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": analyzeSpeechPrompt}],
    temperature=0.5,
)
    
    my_bar.progress(60, text=progress_text)
    
    analysisPrompt = (
        file_contents
        + "\n"
        + completion.choices[0].message.content
        + " Extract the most engaging 25-35 second intervals in [start_time - end_time] format."
    )
    
    completion = openai.ChatCompletion.creat(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": analysisPrompt}],
    )
    
    response = completion.choices[0].message.content.strip().split("\n")
    
    intervals = [list(map(float, line.strip("[]").split(" - "))) for line in response]
    
    my_bar.progress(100, text=progress_text)
    my_bar.empty()
    
    return intervals
