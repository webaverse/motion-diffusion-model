bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh

conda env create
conda env update -n mdm -f environment.yml
conda activate mdm

python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git

cd ..
git clone https://github.com/EricGuo5513/HumanML3D.git
unzip ./HumanML3D/HumanML3D/texts.zip -d ./HumanML3D/HumanML3D/
cp -r HumanML3D/HumanML3D motion-diffusion-model/dataset/HumanML3D
cd motion-diffusion-model

mkdir save
cd save
pip install gdown
gdown 1PE0PK8e5a5j-7-Xhs5YET5U5pGh0c821
unzip humanml_trans_enc_512.zip
cd ..