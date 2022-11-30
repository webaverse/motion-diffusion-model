import asyncio
import json
import os
import sys
from dataclasses import dataclass
from fastapi import FastAPI, Request, BackgroundTasks
import logging
import os
from time import time
import torch
from data_loaders.get_data import get_dataset_loader
from sampler import run
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from visualizer import convertToObj
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware


logging.basicConfig(level=logging.INFO, format="%(levelname)-9s %(asctime)s - %(name)s - %(message)s")
LOGGER = logging.getLogger(__name__)

EXPERIMENTS_BASE_DIR = "/experiments/"
QUERY_BUFFER = {}

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

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*']
)
loop = asyncio.get_event_loop()

@dataclass
class Query():
    query_name: str
    query_sequence: str
    s: str
    gender: str
    isGLTF: str
    result: str = ""
    extention: str = ".glb"
    experiment_id: str = None
    status: str = "pending"

    def __post_init__(self):
        self.experiment_id = str(time())
        self.experiment_dir = os.path.join(EXPERIMENTS_BASE_DIR, self.experiment_id)

@app.get("/generate")
async def root(request: Request, background_tasks: BackgroundTasks, s: str, gender: str, isGLTF: str):
    query = Query(query_name="test", query_sequence=5, s=s, gender=gender, isGLTF=isGLTF)
    QUERY_BUFFER[query.experiment_id] = query
    background_tasks.add_task(process, query)
    LOGGER.info(f'root - added task')
    return {"id": query.experiment_id}

@app.get("/generate_result")
async def result(request: Request, query_id: str):
    print('result')
    if query_id in QUERY_BUFFER:
        if QUERY_BUFFER[query_id].status == "done":
            resp = FileResponse(QUERY_BUFFER[query_id].result, filename="output" + QUERY_BUFFER[query_id].extention)
            del QUERY_BUFFER[query_id]
            return resp
        return {"status": QUERY_BUFFER[query_id].status}
    else:
        return {"status": "finished"}

def process(query):
    LOGGER.info(f"process - {query.experiment_id} - Submitted query job. Now run non-IO work for 10 seconds...")
    start_time = time()
    isGLTF = query.isGLTF == "true"
    extention = ".fbx"
    if isGLTF:
      extention = ".glb"
    
    print(extention, isGLTF)
    path = run(query.s, model, diffusion, state_dict, data)
    basePath = path
    convertToObj(path + "/sample00_rep00.mp4")
    path += "/sample00_rep00_smpl_params.npy.pkl"
    basePath += "/output" + extention
    os.system("python fbx_output.py --input \"" + path + "\" --output \"" + basePath + "\" --fps_source 30 --fps_target 30 --gender " + query.gender + " --person_id 0")

    print("--- %s seconds ---" % (time() - start_time))
    print(basePath, "output" + extention, extention, isGLTF)

    QUERY_BUFFER[query.experiment_id].status = "done"
    QUERY_BUFFER[query.experiment_id].result = basePath
    QUERY_BUFFER[query.experiment_id].extention = extention
    LOGGER.info(f'process - {query.experiment_id} - done!')

@app.get("/backlog/")
def return_backlog():
    return {f"return_backlog - Currently {len(QUERY_BUFFER)} jobs in the backlog."}


if __name__ == "__main__":
    load_model()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7777)