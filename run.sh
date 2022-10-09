#!/bin/bash
python -m sample --model_path ./save/humanml_trans_enc_512/model000200000.pt --num_repetitions 1 --text_prompt "$1"
