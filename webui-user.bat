@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS= --xformers --opt-sdp-no-mem-attention --enable-insecure-extension-access

call webui.bat
