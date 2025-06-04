## Zernike demo

Mini repo to demonstrate the use of the spherical alignment/deform commands in Xmipp. Using a custom python interface module to call various Xmipp commands.

The goal is to iteratively deform an approximated structure by conforming it to projections of the reference structure, each time using the least correlated projection to direct an update. The expected result is end up in the reference structure, which should be a fixed point.

Usage:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python deform.py
```

Reproducing / altering the data can be done with the `prepare.py` script.

