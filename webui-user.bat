@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS= --api --xformers --opt-sdp-no-mem-attention --enable-insecure-extension-access --nowebui

call webui.bat
