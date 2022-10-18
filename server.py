import os
from time import time
import torch
from data_loaders.get_data import get_dataset_loader
from sampler import run
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from visualizer import convertToObj
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

model_path = "./save/humanml_trans_enc_512/model000200000.pt"

class dataArgs:
    def __init__(self, data):
        self.dataset = data['dataset']
        self.latent_dim = data['latent_dim']
        self.layers = data['layers']
        self.arch = data['arch']
        self.cond_mask_prob = data['cond_mask_prob']
        self.emb_trans_dec = data['emb_trans_dec']
        self.noise_schedule = data['noise_schedule']
        self.sigma_small = data['sigma_small']
        self.lambda_vel = data['lambda_vel']
        self.lambda_rcxyz = data['lambda_rcxyz']
        self.lambda_fc = data['lambda_fc']

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*']
)
model = None
diffusion = None
state_dict = None
data = None

def load_model():
    global model, diffusion, state_dict, data
    print("loading model")
    argData = dataArgs({ 'dataset': "humanml", 'latent_dim': 512, 'layers': 8, 'arch': 'trans_enc', 'cond_mask_prob': 0.1, 'emb_trans_dec': False, 'noise_schedule': 'cosine', 'sigma_small': True, 'lambda_vel': 0.0, 'lambda_rcxyz': 0.0, 'lambda_fc': 0.0})
    model, diffusion = create_model_and_diffusion(argData)
    print(f"Loading checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    data = get_dataset_loader(name="humanml",
                                batch_size=1,
                                num_frames=196,
                                split='test',
                                hml_mode='text_only')
    data.fixed_length = 120

@app.get("/generate")
def generate(s: str, gender: str, isGLTF: str):
    start_time = time()
    print(isGLTF)
    isGLTF = isGLTF == "true"
    extention = ".fbx"
    if isGLTF:
      extention = ".glb"
    
    print(extention, isGLTF)
    path = run(s, model, diffusion, state_dict, data)
    basePath = path
    convertToObj(path + "/sample00_rep00.mp4")
    path += "/sample00_rep00_smpl_params.npy.pkl"
    basePath += "/output" + extention
    os.system("python fbx_output.py --input \"" + path + "\" --output \"" + basePath + "\" --fps_source 30 --fps_target 30 --gender " + gender + " --person_id 0")

    print("--- %s seconds ---" % (time() - start_time))
    print(basePath, "output" + extention, extention, isGLTF)
    return FileResponse(basePath, filename="output" + extention)

if __name__ == '__main__':
    load_model()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7777, log_level="debug")