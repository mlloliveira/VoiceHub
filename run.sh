#!/bin/bash
# Configure ASR and TTS languages
export ASR_LANGS=auto,en,es,fr,de,it,pt,pl,tr,ru,nl,cs,ar,zh-cn,ja,hu,ko,hi
export TTS_LANGS=en,es,fr,de,it,pt,pl,tr,ru,nl,cs,ar,zh-cn,ja,hu,ko,hi

# Configure server address/port
export SERVER_NAME=0.0.0.0
export SERVER_PORT=7860

# Run the app
python app.py
