import os, sys, shutil, tarfile
import urllib.request

DATASET="https://storage.googleapis.com/three-cv-research-datasets/cs8680-3dcv/04-reconstruction.tgz"
fname_tgz = os.path.basename(DATASET)

if not os.path.exists(DATASET):
    print('Downloading', fname_tgz)
    with urllib.request.urlopen(DATASET) as response, open(fname_tgz, 'wb') as fout:
        shutil.copyfileobj(response, fout)

if not all(map(os.path.exists, ['Bear', 'Building', 'Deer'])):
    print('Extracting', fname_tgz)
    with tarfile.open(fname_tgz) as tar:
        tar.extractall()

print('Done.')