# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from argparse import ArgumentParser
from datetime import datetime
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from data_loaders.tensors import collate

model_path = "./save/humanml_trans_enc_512/model000200000.pt"



def run(text_prompt, model, diffusion, state_dict, data, seed = 10, dataset = "humanml", motion_length = 6.0, device = 0, guidance_param = 2.5, batch_size = 64, num_repetitions = 1):
    if not os.path.exists("./results"):
        os.makedirs("./results")

    fixseed(seed)
    name = os.path.basename(os.path.dirname(model_path))
    niter = os.path.basename(model_path).replace('model', '').replace('.pt', '')
    out_path = "./results/samples_{}_{}_seed{}_{}".format(name, niter, seed, datetime.now().strftime("%Y%m%d_%H%M%S"))
    max_frames = 196 if dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if dataset == 'kit' else 20
    n_frames = min(max_frames, int(motion_length*fps))
    is_using_data = text_prompt == ''
    dist_util.setup_dist(device)

    if guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    print('Loading dataset...')
    num_samples = 1
    texts = []
    if text_prompt != '':
        texts = [text_prompt]
        
    assert num_samples <= batch_size, \
        f'Please either increase batch_size({batch_size}) or reduce num_samples({num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    batch_size = num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
    total_num_samples = num_samples * num_repetitions

    if is_using_data:
        iterator = iter(data)
        _, model_kwargs = next(iterator)
    else:
        _, model_kwargs = collate(
            [{'inp': torch.tensor([[0.]]), 'target': 0, 'text': txt, 'tokens': None, 'lengths': n_frames}
             for txt in texts]
        )

    all_motions = []
    all_lengths = []
    all_text = []

    for rep_i in range(num_repetitions):
        print(f'### Start sampling [repetitions #{rep_i}]')

        # add CFG scale to batch
        if guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(batch_size, device=dist_util.dev()) * guidance_param

        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(
            model,
            (batch_size, model.njoints, model.nfeats, n_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )


        # Recover XYZ *positions* from HumanML3D vector representation
        if model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        all_text += model_kwargs['y']['text']
        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        print(f"created {len(all_motions) * batch_size} samples")


    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
             'num_samples': num_samples, 'num_repetitions': num_repetitions})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if dataset == 'kit' else paramUtil.t2m_kinematic_chain
    for sample_i in range(num_samples):
        for rep_i in range(num_repetitions):
            caption = all_text[rep_i*batch_size + sample_i]
            length = all_lengths[rep_i*batch_size + sample_i]
            motion = all_motions[rep_i*batch_size + sample_i].transpose(2, 0, 1)[:length]
            save_file = 'sample{:02d}_rep{:02d}.mp4'.format(sample_i, rep_i)
            animation_save_path = os.path.join(out_path, save_file)
            print(f'[({sample_i}) "{caption}" | Rep #{rep_i} | -> {save_file}]')
            if dataset == 'kit':
                motion *= 0.003  # scale for visualization
            elif dataset == 'humanml':
                motion *= 1.3  # scale for visualization
            plot_3d_motion(animation_save_path, skeleton, motion, title=caption, fps=fps)
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')
    return out_path