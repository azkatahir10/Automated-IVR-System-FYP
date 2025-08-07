#!/bin/bash
CALLER_ID=$1
BASE_DIR=/home/asteriskvm/ivr-module

source /home/asteriskvm/ivr-module/venv/bin/activate
python3 $BASE_DIR/main.py $CALLER_ID
