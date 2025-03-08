import streamlit as st
import yt_dlp
import uuid
import psycopg2
import subprocess
from io import BytesIO
from PIL import Image
import re  # For email and phone validation
from datetime import date
from streamlit_option_menu import option_menu
import base64
import whisper
import ffmpeg
import os
import streamlit as st
from textblob import TextBlob
import openai


def download_button(label, file_path, filename):
    with open(file_path, "rb") as file:
        video_bytes = file.read()
        base64_encoded_video = base64.b64encode(video_bytes).decode("utf-8")

    st.markdown(
        f"""
            <a href="data:video/mp4;base64,{base64_encoded_video}" class="button-link" download="{filename}">
                <button class="download-button">
                    {label}
                </button>
            </a>
            """,
        unsafe_allow_html=True,
    )

top_n = 5  # Default number of segments
temperature = 0.5  # Temperature for the GPT model
criteria = "sentiment"  # Default criteria for reel selection (options: "sentiment", "duration", "word_count")

# Step 1: Extract Audio from Video using FFmpeg
def extract_audio(video_path, output_audio_path):
    try:
        ffmpeg.input(video_path).output(output_audio_path).run(overwrite_output=True)
        print(f"Audio extracted successfully to {output_audio_path}")
    except ffmpeg.Error as e:
        print(f"Error extracting audio: {e.stderr.decode() if e.stderr else 'Unknown error'}")

# Step 2: Transcribe Audio to Text using Whisper
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language='en')
    return result['segments']

# Step 3: Analyze Text Segments for Importance
def analyze_text_importance(segments):
    important_segments = []
    for segment in segments:
        text = segment['text']
        start_time = segment['start']
        end_time = segment['end']

        # Perform sentiment analysis using TextBlob
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity  # -1 (negative) to +1 (positive)
        word_count = len(text.split())

        # Score the segment based on sentiment and length
        importance_score = sentiment_score * word_count

        # Only keep segments with a positive importance score
        if importance_score > 0:
            important_segments.append({
                'text': text,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'importance_score': importance_score,
                'sentiment_score': sentiment_score,
                'word_count': word_count
            })

    return important_segments

# Function to use GPT for analysis
def use_gpt_for_analysis(text):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": text}],
        temperature=temperature,
    )
    return response['choices'][0]['message']['content']

# Generate a summary of the video content
def generate_summary(important_segments):
    combined_text = " ".join([segment['text'] for segment in important_segments])
    prompt = f"Summarize the following text: {combined_text}"

    summary = use_gpt_for_analysis(prompt)
    return summary

# Step 4: Select segments to fit within 30 seconds
def select_top_segments(important_segments,criteria,total_duration=30):
    if criteria == 'sentiment':
        important_segments.sort(key=lambda x: x['sentiment_score'], reverse=True)
    elif criteria == 'duration':
        important_segments.sort(key=lambda x: x['duration'], reverse=True)
    elif criteria == 'word_count':
        important_segments.sort(key=lambda x: x['word_count'], reverse=True)

    selected_segments = []
    current_duration = 0

    for segment in important_segments:
        if current_duration + segment['duration'] <= total_duration:
            selected_segments.append(segment)
            current_duration += segment['duration']

    return selected_segments

# Step 5: Extract and Concatenate Video Segments into a Single File
def extract_and_concat_video_segments(video_path, segments, output_video_path):
    segment_files = []
    for i, segment in enumerate(segments):
        start_time = segment['start_time']
        duration = segment['duration']
        output_segment_path = f'segment_{i}.mp4'

        try:
            ffmpeg.input(video_path, ss=start_time, t=duration).output(
                output_segment_path,
                vf="scale=854:480:force_original_aspect_ratio=decrease,pad=854:480:(ow-iw)/2:(oh-ih)/2"
).run(overwrite_output=True)
            segment_files.append(output_segment_path)
        except ffmpeg.Error as e:
            print(f"Error extracting video segment: {e.stderr.decode() if e.stderr else 'Unknown error'}")

    # Concatenate video segments
    with open('file_list.txt', 'w') as f:
        for segment in segment_files:
            f.write(f"file '{segment}'\n")

    try:
        ffmpeg.input('file_list.txt', format='concat', safe=0).output(
            output_video_path, c='copy'
        ).run(overwrite_output=True)
        print(f"Compiled video (16:9) created: {output_video_path}")
    except ffmpeg.Error as e:
        print(f"Error concatenating video: {e.stderr.decode() if e.stderr else 'Unknown error'}")
    finally:
        os.remove('file_list.txt')
        for segment in segment_files:
            os.remove(segment)

# Full Process: Text Analysis, Timestamp Mapping, Video Extraction, and Compilation
def generate_30sec_reel(video_path, audio_path, compiled_video_path):
    extract_audio(video_path, audio_path)

    segments, subtitle_path = transcribe_audio(audio_path)
    important_segments = analyze_text_importance(segments)

    selected_segments = select_top_segments(important_segments, criteria, total_duration=30)
    extract_and_concat_video_segments(video_path, selected_segments, compiled_video_path)

    final_output = compiled_video_path.replace(".mp4", "_subtitled.mp4")

    # Add subtitles to the video
    add_subtitles(compiled_video_path, subtitle_path, final_output)

    return final_output, generate_summary(important_segments)

def add_subtitles(video_path, subtitle_path, output_path):
    try:
        ffmpeg.input(video_path).output(output_path, vf=f"subtitles={subtitle_path}").run(overwrite_output=True)
        print(f"Subtitled video created: {output_path}")
    except ffmpeg.Error as e:
        print(f"Error adding subtitles: {e.stderr.decode() if e.stderr else 'Unknown error'}")


def extract_audio(video_path, audio_path):
    command = ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_path]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            st.error(f"Error extracting audio: {result.stderr.decode()}")
            return False
        if not os.path.exists(audio_path):
            st.error(f"Audio file not created: {audio_path}")
            return False
        return True
    except Exception as e:
        st.error(f"Audio extraction failed: {e}")
        return False
def cleanup_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                st.error(f"Error deleting file: {file_path}. Details: {e}")

# Function to call the reel generation process and handle summary display
def generate_reel_and_summary(video_file):
    unique_id = str(uuid.uuid4())
    video_path = f'uploaded_video_{unique_id}.mp4'
    audio_path = f'output_audio_{unique_id}.wav'
    compiled_video_paths = []

    try:
        # Save the uploaded video file
        with open(video_path, 'wb') as f:
            f.write(video_file.read())

        # Step 1: Extract audio from the video
        if not extract_audio(video_path, audio_path):
            cleanup_files([video_path])
            return None, None

        # Step 2: Transcribe audio to text
        segments = transcribe_audio(audio_path)

        # Step 3: Analyze text segments to determine importance
        important_segments = analyze_text_importance(segments)

        # Step 4: Generate a summary of the important segments
        video_summary = generate_summary(important_segments)

        # Step 5: Generate and save the reel for each criterion
        criteria_list = ['sentiment', 'duration', 'word_count']
        for criteria in criteria_list:
            compiled_video_path = f'compiled_reel_{unique_id}_{criteria}.mp4'
            # Pass the criteria when selecting top segments
            selected_segments = select_top_segments(important_segments, criteria=criteria, total_duration=35)  # Adjust duration if needed
            extract_and_concat_video_segments(video_path, selected_segments, compiled_video_path)
            compiled_video_paths.append((compiled_video_path, criteria))

        cleanup_files([video_path, audio_path])

        # Step 6: Return the generated reels and the video summary
        reel_data = []
        for reel_path, criteria in compiled_video_paths:
            if os.path.exists(reel_path):
                with open(reel_path, 'rb') as reel_file:
                    reel_data.append((reel_file.read(), criteria, reel_path))
                os.remove(reel_path)

        return reel_data, video_summary

    except Exception as e:
        st.error(f"Error generating reel: {str(e)}")
        cleanup_files([video_path, audio_path])
        return None, None

DOWNLOAD_PATH = "C:\\Users\\ASUS\\Desktop\\v_r youtube download"
def download_youtube_video(url):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',  # Best video and audio in MP4 format
        'outtmpl': os.path.join(DOWNLOAD_PATH, '%(title)s.%(ext)s'),  # Save the video with title as filename
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            # Get the prepared filename for the video
            video_info = ydl.extract_info(url, download=False)
            filename = ydl.prepare_filename(video_info)
        video_path = os.path.join(DOWNLOAD_PATH, filename)

        # Check the file size (in bytes)
        file_size = os.path.getsize(video_path)

        # If file size is greater than 200 MB (200 * 1024 * 1024 bytes)
        if file_size > 200 * 1024 * 1024:  # 200 MB
            return f"File size is too large for generating a reel. The video size is {file_size / (1024 * 1024):.2f} MB."

        return video_path
    except Exception as e:
        return f"An error occurred: {e}"

def reel_generation_page():
    try:
        if "active_tab" not in st.session_state:
            st.session_state.active_tab = "Upload a video file"
        if "video_path" not in st.session_state:
            st.session_state.video_path = None
        if "youtube_url" not in st.session_state:
            st.session_state.youtube_url = ""
        if "reel_data" not in st.session_state:
            st.session_state.reel_data = None
        if "summary_text" not in st.session_state:
            st.session_state.summary_text = None
        # Tabs for "Upload a video file" and "Enter YouTube URL"
        tab_upload, tab_youtube = st.tabs(["Upload a video file", "Enter YouTube URL"])

        # Handle tab switching and clear data when switching
        with tab_upload:
            # Set active tab
            st.session_state.active_tab = "Upload a video file"
            st.session_state.youtube_url = ""

            # Reset data for this tab
            if st.session_state.video_path is not None:
                st.session_state.video_path = None  # Reset video path when switching tabs

            # Handle video file upload
            uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
            if uploaded_file is not None:
                st.session_state.video_path = uploaded_file  # Store the uploaded file path in session state
                st.video(uploaded_file)  # Display the uploaded video

            # Generate reel button
            if st.session_state.video_path is not None and st.button("Generate Reel 🎬"):
                with st.spinner("Snails are nature’s slowpokes! They carry their homes on their backs, but still manage to be the last ones to arrive at any party — no RSVP needed, they’re always fashionably late!⏳"):
                    reel_data, summary_text = generate_reel_and_summary(st.session_state.video_path)
                    st.session_state.reel_data = reel_data
                    st.session_state.summary_text = summary_text

        with tab_youtube:
            # Set active tab
            st.session_state.active_tab = "Enter YouTube URL"
            st.session_state.video_path = None

            # Reset data for this tab
            if st.session_state.video_path is not None:
                st.session_state.video_path = None  # Reset video path when switching tabs

            # Handle YouTube URL input
            youtube_url = st.text_input("Enter YouTube video URL")
            if youtube_url and st.button("URL Given 🎬"):
                with st.spinner("Waiting time is like a magician’s trick: the longer you wait, the more your mind starts creating impossible scenarios — like how the pizza guy might’ve been abducted by aliens, or how the elevator could be taking a detour to the moon!"):
                    video_path = download_youtube_video(youtube_url)
                    st.session_state.video_path = video_path  # Store the downloaded video path in session state
                if video_path:
                    st.video(video_path)
                    with open(video_path, "rb") as video_file:
                        with st.spinner("Time spent waiting is never wasted — it's simply training you for the ultimate test of patience… until you finally get distracted and forget what you were waiting for in the first place!⏳"):
                            reel_data, summary_text = generate_reel_and_summary(video_file)
                        st.session_state.reel_data = reel_data
                        st.session_state.summary_text = summary_text
        if st.session_state.reel_data:
            # Initialize the selected reel in session_state if it doesn't exist
            if "selected_reel" not in st.session_state:
                st.session_state.selected_reel = None

            # Create columns for the reels (Grid layout)
            num_cols = 3  # Adjust this to the desired number of columns
            cols = st.columns(num_cols)

            for index, (reel, criteria, reel_path) in enumerate(st.session_state.reel_data):
                # Determine the column for the reel
                col_index = index % num_cols
                with cols[col_index]:
                    # Display a thumbnail or small preview and a select button
                    st.video(reel, format="video/mp4") # Changed this too

                    if st.button(f"View Reel {index + 1} (Criteria: {criteria})", key=f"view_button_{index}"):
                        st.session_state.selected_reel = (reel, criteria, reel_path)  # Store the selected reel

            # Display the selected reel if available
            if st.session_state.selected_reel:
                st.subheader("Selected Reel - Enlarged View")
                st.video(st.session_state.selected_reel[0], format="video/mp4")

                # Download button for the selected reel
                st.download_button(
                    label="Download Selected Reel",
                    data=st.session_state.selected_reel[0],
                    file_name=os.path.basename(st.session_state.selected_reel[2]),
                    mime="video/mp4",
                )
            # Display video summary if available
            if st.session_state.summary_text:
                st.subheader("Video Summary 📝")
                st.markdown(f"*Summary*:\n{st.session_state.summary_text}")
            else:
                st.error("Error generating reel. Please try again. ❌")
    except Exception as e:
        st.error(f"Error generating reel: {str(e)}")