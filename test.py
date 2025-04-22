import torch as th
import blobfile as bf
import os
from improved_diffusion import logger

def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())

print("start")
with bf.BlobFile(bf.join(get_blob_logdir(), "model111.pt"), "wb") as f:
    print(f)