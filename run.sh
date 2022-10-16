#!/bin/bash
# check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "conda could not be found"
    exit
fi

# ./fbx202001_fbxpythonsdk_linux .

# conda env create
# conda env update -n mdm -f environment.yml
# conda activate mdm
    
python3 server.py