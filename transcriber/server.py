#!/usr/bin/env python3

# pip install git+https://github.com/m-bain/whisperx.git aiohttp

import whisperx
import argparse
import io
import asyncio
from aiohttp import web
import tempfile
import logging
import concurrent.futures
import json

logger = logging.getLogger(__name__)

def transcribe(audio_file: str, model: str = "small", device: str = "cpu", compute_type: str = "int8"):
    batch_size = 16  # reduce if low on GPU mem

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model(model, device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    result.update(aligned_result)
    return result

def get_vtt(result: dict):
    with io.StringIO() as stream:
        writer = whisperx.utils.WriteVTT(".")
        writer.write_result(result, stream, {"max_line_width": 32, "max_line_count": 2, "highlight_words": True})
        return stream.getvalue()

def get_text(result: dict) -> str:
    with io.StringIO() as stream:
        writer = whisperx.utils.WriteTXT(".")
        writer.write_result(result, stream, {})
        return stream.getvalue()

def transcribe2(audio_file_path: str):
    result = transcribe(audio_file_path)
    response = result
    response['vtt'] = get_vtt(result)
    response['text'] = get_text(result)
    return response

async def api_transcribe(request):
    data = await request.post()
    audio_file = data['audio_file'].file
    af = tempfile.NamedTemporaryFile(mode="wb")
    try:
        af.write(audio_file.read())
        af.flush()
        executor = request.app['thread_executor']
        logger.info("Runnign transcribe task in thradpool executor, file %s", af.name)
        response = await asyncio.get_event_loop().run_in_executor(executor, transcribe2, af.name)
        return web.json_response(response)
    finally:
        af.close()

async def transcription_api():
    app = web.Application(client_max_size=1024**4)
    app.router.add_post('/transcribe', api_transcribe)
    app['thread_executor'] = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    return app

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--audio-file", type=str, help="Do not run server, use the audio file to transcode and output the result to the stdout")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    if args.audio_file:
        result = transcribe2(args.audio_file)
        print(json.dumps(result, indent=4))
    else:
        loop = asyncio.get_event_loop()
        app = loop.run_until_complete(transcription_api())
        web.run_app(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()