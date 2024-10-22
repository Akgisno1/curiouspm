import streamlit as st
import io
import os
import moviepy.editor as mp
from google.cloud import speech_v1 as speech
from google.cloud import texttospeech_v1 as texttospeech
from pydub import AudioSegment
import requests
import re

azure_openai_key = "22ec84421ec24230a3638d1b51e3a7dc"
azure_openai_endpoint = "https://internshala.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"credentials.json"

def transcribe_audio(audio_file):
    client = speech.SpeechClient()
    audio_segment = AudioSegment.from_wav(audio_file)
    audio_segment = audio_segment.set_channels(1)
    audio_segment = audio_segment.set_frame_rate(16000)
    mono_audio_file = "temp_mono_audio.wav"
    audio_segment.export(mono_audio_file, format="wav")

    with io.open(mono_audio_file, "rb") as audio:
        content = audio.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )
    
    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=90)

    transcription = []
    for result in response.results:
        transcription.append(result.alternatives[0].transcript)

    return " ".join(transcription)

def clean_transcription(transcription):
    cleaned_text = re.sub(r'\b(um|uh|like|you know|so|actually|basically)\b', '', transcription, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text.strip()

def correct_transcription(transcription):
    headers = {
        "Content-Type": "application/json",
        "api-key": azure_openai_key
    }
    data = {
        "messages": [
            {
                "role": "user",
                "content": f"Correct the following transcription, removing any filler words: '{transcription}'"
            }
        ],
        "max_tokens": 100
    }

    response = requests.post(azure_openai_endpoint, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    else:
        st.error(f"Error in OpenAI API call: {response.status_code}")
        return None

def synthesize_speech(text, video_duration):
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Wavenet-D"
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        speaking_rate=0.9  # Slow down the speech rate (default is 1.0)
    )

    response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)

    output_audio_file = "output_audio.wav"
    with open(output_audio_file, "wb") as out:
        out.write(response.audio_content)
    
    # Adjust length of the audio to match video duration
    adjust_audio_to_video_duration(output_audio_file, video_duration)
    
    return output_audio_file

def adjust_audio_to_video_duration(audio_file, video_duration):
    audio = AudioSegment.from_wav(audio_file)
    video_duration_ms = video_duration * 1000  # Convert to milliseconds

    if len(audio) < video_duration_ms:
        silence_duration = video_duration_ms - len(audio)
        silence = AudioSegment.silent(duration=silence_duration)
        final_audio = audio + silence  # Append silence to the end of the audio
    else:
        # Keep the audio longer and do not trim
        final_audio = audio  # Do not trim the audio

    final_audio.export(audio_file, format="wav")

def replace_audio_in_video(video_file, audio_file):
    video = mp.VideoFileClip(video_file)
    audio = mp.AudioFileClip(audio_file)
    final_video = video.set_audio(audio)
    final_video_file = "final_video.mp4"
    final_video.write_videofile(final_video_file, codec='libx264', audio_codec='aac')
    return final_video_file

def main():
    st.title("Video Audio Replacement with AI")
    st.markdown("### Please ensure the video is NOT more than 1 min of length for optimal audio replacement.")

    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov"])
    
    if video_file is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(video_file.getbuffer())

        video = mp.VideoFileClip("temp_video.mp4")
        audio_path = "temp_audio.wav"
        video.audio.write_audiofile(audio_path)

        transcription = transcribe_audio(audio_path)
        cleaned_transcription = clean_transcription(transcription)
        corrected_transcription = correct_transcription(cleaned_transcription)

        if corrected_transcription:
            new_audio_file = synthesize_speech(corrected_transcription, video.duration)
            final_video_file = replace_audio_in_video("temp_video.mp4", new_audio_file)
            st.video(final_video_file)

if __name__ == "__main__":
    main()
