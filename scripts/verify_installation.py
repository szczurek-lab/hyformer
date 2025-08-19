""" Verify the installation of the Hyformer package.
"""

import sys
import logging
import torch


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)


def main():
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Number of recognized GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logger.info(f'CUDA:{i}: {torch.cuda.get_device_properties(i).name}')

    try: 
        import hyformer
    except ImportError as e:
        logger.error(f"Hyformer installation not found: {e}")
    else:
        logger.info(f"Hyformer installation with version {hyformer.__version__} found")

if __name__ == "__main__":
    main()
