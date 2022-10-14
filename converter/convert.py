from converter.fbxWriter import FbxReadWrite
from converter.smplObject import SmplObjects
import tqdm
import traceback

def convert(path, basePath):
    fbx_source_path = "./fbx/SMPL_m_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx"
    obj = SmplObjects(path)
    print("loaded")
    for pkl_name, smpl_params in tqdm.tqdm(obj):
        try:
            fbx = FbxReadWrite(fbx_source_path)
            fbx.addAnimation(pkl_name, smpl_params)
            fbx.writeFbx(basePath)
        except Exception as e:
            print("Exception: {}".format(e))
            traceback.print_exc()
        finally:
            fbx.destroy()