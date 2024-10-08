# Transcriber

A Python package for transcribing audio and video files using WhisperX and SpeechRecognition.

## Installation

```sh
pip install git+https://github.com/dzianisv/transcriber.git

```

## Run a transcriber server

```sh
python3 -m transcriber.server --host 100.100.107.153 --port 8083
```


## Transcriber the video file
```sh
python3 -m transcriber.client video.mp4
```
