#!/usr/bin/env python3

import unittest
import subprocess
import time
from aiohttp import ClientSession
from contextlib import asynccontextmanager

# Sample audio URL (ensure you have the rights to use it)
sample_audio_url = "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"

class TestTranscriber(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # Start the transcriber server
        cls.server_process = subprocess.Popen(['python3', 'server.py'])
        time.sleep(5)  # Wait for the server to start

    @classmethod
    def tearDownClass(cls):
        # Terminate the server process
        cls.server_process.terminate()
        cls.server_process.wait()

    async def test_transcription(self):
        async with ClientSession() as session:
            # Download the sample audio file
            async with session.get(sample_audio_url) as response:
                audio_data = await response.read()

            # Send the audio file to the transcriber server
            async with session.post("http://localhost:8080/transcribe", data={'audio_file': audio_data}) as response:
                self.assertEqual(response.status, 200)
                response_json = await response.json()
                transcription_text = response_json['text']
                print("Transcription:", transcription_text)

                # Basic check to ensure transcription is not empty
                self.assertTrue(transcription_text)

if __name__ == '__main__':
    unittest.main()