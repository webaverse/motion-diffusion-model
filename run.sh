#!/bin/bash
if ! command -v conda &> /dev/null
then
    echo "conda could not be found, please run install.sh first"
    exit
fi

python3 server.py