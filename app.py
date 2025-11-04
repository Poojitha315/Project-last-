from flask import Flask, request, render_template, jsonify
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from transformers import pipeline
import os
from pydub import AudioSegment
from pydub.utils import make_chunks

app = Flask(__name__)

# Load summarization model (offline model, no API key needed)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    video = request.files['video']
    os.makedirs("uploads", exist_ok=True)
    video_path = os.path.join("uploads", video.filename)
    video.save(video_path)

    # Extract audio from video
    clip = VideoFileClip(video_path)
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    clip.audio.write_audiofile(audio_path)

    # Convert speech to text in chunks
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_wav(audio_path)
    chunk_length_ms = 60000  # 60 seconds
    chunks = make_chunks(audio, chunk_length_ms)

    full_text = ""
    for i, chunk in enumerate(chunks):
        chunk_filename = f"chunk_{i}.wav"
        chunk.export(chunk_filename, format="wav")

        with sr.AudioFile(chunk_filename) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
                full_text += text + " "
            except sr.UnknownValueError:
                print(f"Chunk {i}: Could not understand audio")
            except sr.RequestError as e:
                print(f"Chunk {i}: API error - {e}")

        os.remove(chunk_filename)

    if not full_text.strip():
        return jsonify({'error': 'Could not transcribe audio'}), 400

    # Summarize the full text
    summary = summarizer(full_text, max_length=200, min_length=60, do_sample=False)[0]['summary_text']

    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
