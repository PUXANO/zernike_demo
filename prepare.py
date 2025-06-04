'''
Script to create a helix, its cartesian 3 projections, and an approximated structure.

'''

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import pandas as pd

from tools.xmipp import Xmipp, xmd_write

cwd = Path(__file__).parent / "prepared"
cwd.mkdir(exist_ok=True)

zs = np.linspace(-30,30,601)
ts = np.linspace(0,2 * np.pi * 4,601)
reference_coordinates = np.stack([zs,10 * np.cos(ts),20 * np.sin(ts)],-1)
small_rot_X = Rotation.from_rotvec([0.055,0.0,0.0]).as_matrix()
approximated_coordinates = reference_coordinates @ small_rot_X

xmipp = Xmipp(cwd)

# create volumes
reference_path = xmipp.volume_from_pdb(reference_coordinates,'reference')
approximated_path = xmipp.volume_from_pdb(approximated_coordinates,'approximation')

#create projection
selection = pd.DataFrame([[0.0,  0.0,  0.0, 1, 0.0, 0.0],
                          [0.0,  0.0, 90.0, 1, 0.0, 0.0],
                          [0.0, 90.0, 90.0, 1, 0.0, 0.0]],
                         columns = ['anglePsi','angleRot','angleTilt','enabled','shiftX','shiftY'],
                         index = ['XY','ZY','ZX'])
xmd_write(selection,angles_path := reference_path.with_suffix('.doc'))
reference_projections, reference_directions = xmipp.angular_project_selection(reference_path,angles_path)

