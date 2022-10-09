#!/bin/bash
python -m sample --model_path ./save/humanml_trans_enc_512/model000475000.pt --num_repetitions 1 --text_prompt "$1"
