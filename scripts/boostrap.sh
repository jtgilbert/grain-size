#! /bin/bash

set -eu

python3 --version
python3 -m venv ../.venv

../.venv/bin/python -m pip install --upgrade pip

../.venv/bin/pip install -e ../