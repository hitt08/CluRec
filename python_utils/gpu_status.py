import logging
def check_cupy_gpu():
    import cupy as cp
    log = logging.getLogger(__name__)
    log.warning("=" * 10 + "CuPy" + "=" * 10)
    log.warning(f"  Version: {cp.__version__}")
    log.warning(f"  GPU Available: {cp.cuda.is_available()}")
    log.warning("=" * 26)
    log.warning("")

def check_torch_gpu():
    import torch
    log = logging.getLogger(__name__)
    log.warning("=" * 8 + " pyTorch " + "=" * 9)
    log.warning(f"  Version: {torch.__version__}")
    log.warning(f"  Number of GPUs Available: {torch.cuda.device_count()}")
    log.warning(f"  Compiled for CUDA Architectures: {torch.cuda.get_arch_list()}")
    log.warning("=" * 26)
    log.warning("")

def check_tf_gpu():
    import tensorflow as tf
    log = logging.getLogger(__name__)
    log.warning("=" * 7 + " TensorFlow " + "=" * 7)
    log.warning(f"  Version: {tf.__version__}")
    log.warning(f"  Number of GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
    log.warning("=" * 26)
    log.warning("")
