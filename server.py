from flask import Flask, request, send_file
import torch
from converter.convert import convert
from data_loaders.get_data import get_dataset_loader
from sampler import run
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from visualizer import convertToObj

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

app = Flask(__name__)
model = None
diffusion = None
state_dict = None
data = None

def mkResponse(data):
  return send_file(
    data,
    download_name="image.png",
    mimetype="image/png",
  )

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

@app.route('/generate', methods=['GET'])
def generate():
    s = request.args.get("s")
    path = run(s, model, diffusion, state_dict, data)
    basePath = path
    convertToObj(path + "/sample00_rep00.mp4")
    path += "/sample00_rep00_smpl_params.npy.pkl"
    convert(path, basePath)
    basePath += "/output.fbx"

    response = send_file(basePath, download_name="output.fbx", mimetype="application/octet-stream")
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    return response

if __name__ == '__main__':
    load_model()
    app.run(host="0.0.0.0", port=7777)