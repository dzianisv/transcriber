#!/usr/bin/env python3

import os
import sys
import asyncio
import logging
import aiohttp
from contextlib import asynccontextmanager
from typing import Optional, Tuple, AsyncGenerator


logger = logging.getLogger(__name__)

# Function to extract audio from video using ffmpeg
@asynccontextmanager
async def extract_audio(video_file):
    audio_file = video_file.rsplit('.', 1)[0] + '.opus'
    command = ["ffmpeg", "-i", video_file, "-q:a", "0", "-map", "a", "-c:a", "libopus", "-b:a", "64k", audio_file]
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        _stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise Exception(f"Command failed with return code {process.returncode}\n{stderr.decode()}")
        else:
            yield audio_file
    finally:
        if os.path.exists(audio_file):
            os.remove(audio_file)

async def transcribe_audio(audio_path: str, endpoint: Optional[str]) -> Optional[str]:
    try:
        if endpoint is None:
            raise Exception("WhisperX endpoint not set")

        async with aiohttp.ClientSession() as session:
            with open(audio_path, 'rb') as audio_file:
                data = aiohttp.FormData()
                data.add_field('audio_file', audio_file)
                
                logger.info("Using WhisperX endpoint: %s", endpoint)
                async with session.post(f"{endpoint}/transcribe", data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['text']
                    response.raise_for_status()
                    
    except (aiohttp.ClientError, KeyError) as e:
        logger.warning("WhisperX service unavailable, falling back to Google Speech API: %s", str(e))
        import speech_recognition as sr
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
            
        try:
            return recognizer.recognize_google(audio)
        except sr.RequestError as e:
            logger.error("Google Speech API error: %s", str(e))
            raise
        except sr.UnknownValueError:
            logger.error("Google Speech API could not understand the audio")
            raise

async def transcribe_video(video_file: str, endpoint: Optional[str]) -> Optional[str]:
    async with extract_audio(video_file) as audio_file:
        try:
            logger.info("Transcribing %s", audio_file)
            transcription = await transcribe_audio(audio_file)
            logger.info(f"Transcription: {transcription}")
            return transcription
        except Exception as e:
            logger.exception(e)
            return ""

if __name__ == "__main__":
    endpoint = os.environ.get("endpoint", None)
    if not endpoint:
        whisperx_host = os.environ.get("WHISPERX_HOST", "localhost")
        whisperx_port = os.environ.get("WHISPERX_PORT", "8080")
        endpoint = f"http://{whisperx_host}:{whisperx_port}"

    logging.basicConfig(level=logging.INFO)
    asyncio.run(transcribe_video(sys.argv[1], endpoint))