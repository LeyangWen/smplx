import pickle
import os.path as osp
from typing import Union

import numpy as np
from human_body_prior.tools.omni_tools import get_support_data_dir

# SMPLX to SMPL
# copy and replace moshpp/src/moshpp/marker_layout/marker_vids.py
# custom marker vids for VEHS-7M markers


# run in soma project

def smplx2smplh(vids: Union[list, str]) -> Union[list, str]:
    support_dir = get_support_data_dir(__file__)

    smplx2smplh = np.load(osp.join(support_dir, 'smplx_fit2_smplh.npz'))['smhf2smh']

    if isinstance(vids, int):
        return int(smplx2smplh[vids])
    return [int(smplx2smplh[vid]) for vid in vids]

all_marker_vids = {'smpl':{}, 'smplh':{},
    'smplx': {
                        'HDTP':	    9011,
                        'REAR':	    1050,
                        'LEAR':	    560,
                        'MDFH':	    8949,
                        'C7':	    3353,
                        'C7_d':	    5349,
                        'SS':	    5533,
                        'T8':	    5487,
                        'XP':	    5534,
                        'RPSIS':	    7141,
                        'RASIS':	    8421,
                        'LPSIS':	    4405,
                        'LASIS':	    5727,
                        'RAP':	    6175,
                        'RAP_b':	    6632,
                        'RAP_f':	    7253,
                        'LAP':	    3414,
                        'LAP_b':	    4431,
                        'LAP_f':	    4517,
                        'RLE':	    6695,
                        'RME':	    7107,
                        'LLE':	    4251,
                        'LME':	    4371,
                        'RRS':	    7462,
                        'RUS':	    7458,
                        'LRS':	    4726,
                        'LUS':	    4722,
                        'RMCP2':	    7483,
                        'RMCP5':	    7525,
                        'LMCP2':	    4747,
                        'LMCP5':	    4788,
                        'RIC':	    7149,
                        'RGT':	    6831,
                        'LIC':	    4413,
                        'LGT':	    4088,
                        'RMFC':	    6747,
                        'RLFC':	    6445,
                        'LMFC':	    3999,
                        'LLFC':	    3684,
                        'RMM':	    8680,
                        'RLM':	    8576,
                        'LMM':	    8892,
                        'LLM':	    5882,
                        'RMTP1':	    8587,
                        'RMTP5':	    8593,
                        'LMTP1':	    5893,
                        'LMTP5':	    5899,
                        'RHEEL':	    8634,
                        'LHEEL':	    8846}}


all_marker_vids['smplh'].update(
    {k: smplx2smplh(v) for k, v in all_marker_vids['smplx'].items()})
all_marker_vids['smpl'] = all_marker_vids['smplh']
print(all_marker_vids)

marker_type_labels = {
    'wrist': [
        "RRS",
        "RUS",
        "LRS",
        "LUS"

    ],
    'finger_left': [
        "LMCP2",
        "LMCP5"


    ],
    'finger_right': [
        "RMCP2",
        "RMCP5"

    ],
    'face': [
        "HDTP",
        "REAR",
        "LEAR",
        "MDFH"

    ]
}


with open("marker_vids.pkl", "wb") as f:
    pickle.dump(all_marker_vids, f)