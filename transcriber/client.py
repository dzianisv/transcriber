
import os
import asyncio
import aiohttp
from urllib.parse import urlparse
import logging
from contextlib import asynccontextmanager
from typing import Optional, Tuple, AsyncGenerator


whisperx_endpoint = os.environ.get("WHISPERX_ENDPOINT", "http://localhost:8080")
logger = logging.getLogger(__name__)

# Function to extract audio from video using ffmpeg
@asynccontextmanager
async def extract_audio(video_file) -> AsyncGenerator[str]:
    audio_file = video_file.rsplit('.', 1)[0] + '.m4a'
    command = ["ffmpeg", "-i", video_file, "-q:a", "0", "-map", "a", audio_file]
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise Exception(f"Command failed with return code {process.returncode}\n{stderr.decode()}")
        else:
            yield audio_file
    finally:
        os.remove(audio_file)

async def transcribe_audio(audio_path: str) -> AsyncGenerator[str]:
    try:
        async with aiohttp.ClientSession() as session:
            with open(audio_path, 'rb') as audio_file:
                files = {'audio_file': audio_file.read()}
                async with session.post(f"{whisperx_endpoint}/transcribe", data=files) as response:
                    response.raise_for_status()
                    response_json = await response.json()
                    return response_json['text']
    except Exception as e:
        logger.error("WhisperX failed to transcribe the audio, trying the SpeechRecognition lib as a fallback...")
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)

async def transcribe_video(video_file: str) -> AsyncGenerator[str]:
    async with extract_audio(video_file) as audio_file:
        try:
            logger.info("Transcribing %s", audio_file)
            transcription = await transcribe_audio(audio_file)
            logger.info(f"Transcription: {transcription}")
            return transcription
        except Exception as e:
            logger.exception(e)
            return ""
