import argparse
import os
from visualize import vis_utils
import shutil
from tqdm import tqdm

def convertToObj(input_path, device = '0', cuda = True):
    assert input_path.endswith('.mp4')
    parsed_name = os.path.basename(input_path).replace('.mp4', '').replace('sample', '').replace('rep', '')
    sample_i, rep_i = [int(e) for e in parsed_name.split('_')]
    npy_path = os.path.join(os.path.dirname(input_path), 'results.npy')
    out_npy_path = input_path.replace('.mp4', '_smpl_params.npy')
    assert os.path.exists(npy_path)
    results_dir = input_path.replace('.mp4', '_obj')
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    npy2obj = vis_utils.npy2obj(npy_path, sample_i, rep_i,
                                device=device, cuda=cuda)

    print('Saving obj files to [{}]'.format(os.path.abspath(results_dir)))
    for frame_i in tqdm(range(npy2obj.real_num_frames)):
        npy2obj.save_obj(os.path.join(results_dir, 'frame{:03d}.obj'.format(frame_i)), frame_i)

    print('Saving SMPL params to [{}]'.format(os.path.abspath(out_npy_path)))
    npy2obj.save_npy(out_npy_path)
