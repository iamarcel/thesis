# coding=utf-8
from __future__ import print_function
import json
import os
import os.path
from watson_developer_cloud import TextToSpeechV1

if ('WATSON_TTS_USERNAME' not in os.environ or
    'WATSON_TTS_PASSWORD' not in os.environ):
  print("Please specify your API credentials by setting the environment variables")
  print("- WATSON_TTS_USERNAME")
  print("- WATSON_TTS_PASSWORD")
  exit(1)

tts_username = os.environ['WATSON_TTS_USERNAME']
tts_password = os.environ['WATSON_TTS_PASSWORD']

text_to_speech = TextToSpeechV1(
    username=tts_username,
    password=tts_password)

def write_tts_clip(filename, text):
  with open(filename, 'wb') as audio_file:
      audio_file.write(
          text_to_speech.synthesize(text, accept='audio/wav',
                                    voice="en-US_AllisonVoice").content)
