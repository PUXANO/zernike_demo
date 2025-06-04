'''
Script to iteratively deform an approximated structure by conforming it to projections of the reference structure, 
each time using the least correlated projection to direct an update. 

The expected result is to end up in the reference structure, which should be a fixed point.
'''

from pathlib import Path

import starfile

from tools.xmipp import Xmipp

prepared = Path(__file__).parent / "data"
cwd = Path(__file__).parent / "cwd"

xmipp = Xmipp(cwd)

approximated_coordinates = prepared / "approximation.pdb"
reference_directions = prepared / "reference.doc"

for i in range(20):
    approximated_path = xmipp.volume_from_pdb(approximated_coordinates,'approximated')
    res = xmipp.get_deformations(reference_directions,approximated_path)
    xmd = starfile.read(str(res)).assign(plane = ['XY','ZY','ZX']).set_index('plane')
    worst_plane = xmd.cost.idxmin()
    coef = xmd.sphCoefficients[worst_plane]
    approximated_coordinates = xmipp.apply_deformations(approximated_coordinates,coef,label=f'approximation_{i:02d}')
    print(f'Correlations - XY: {xmd.cost.XY:.03f} - ZY: {xmd.cost.ZY:.03f} - ZX: {xmd.cost.ZX:.03f} - Correcting for {worst_plane} plane')
