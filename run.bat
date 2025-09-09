@echo off
REM Configure ASR and TTS languages
set ASR_LANGS=auto,en,es,fr,de,it,pt,pl,tr,ru,nl,cs,ar,zh-cn,ja,hu,ko,hi
set TTS_LANGS=en,es,fr,de,it,pt,pl,tr,ru,nl,cs,ar,zh-cn,ja,hu,ko,hi

REM Configure server address/port
set SERVER_NAME=127.0.0.1
set SERVER_PORT=7870

REM Run the app
python app.py
pause