# How to run the script?

1. Install detectron2 by following the instructions here: https://detectron2.readthedocs.io/en/latest/tutorials/install.html
You need to make sure that you have CUDA, torch and torchvision installed, as well as the correct python version configured. Afterwards, you can simply run this to install detectron2: `python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'`

2. Install the progress bar: `pip install tqdm`

3. Run the `create_masks.py` script by passing the image directory. The script will create the masks inside the given directory for each image.