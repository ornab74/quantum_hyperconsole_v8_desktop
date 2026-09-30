#!/usr/bin/env sh
set -eu
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
python -m pip check
