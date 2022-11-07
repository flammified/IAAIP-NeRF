<p align="center">
  <img src="https://d2k0ddhflgrk1i.cloudfront.net/Websections/Huisstijl/Bouwstenen/Logo/02-Visual-Bouwstenen-Logo-Varianten-v1.png"/><br>
  <a href="">Report</a>
  <br><br>
</p>

# Rendering forward facing driving scenes using Neural Radiance Fields

This repository contains the code for the Interdisciplinary Advanced AI Project (IAAIP) of the TU Delft. During this project we have worked on `Rendering forward facing driving scenes using Neural Radiance Fields`. The project took 10 weeks in total.

The repository structure is as follows:

- `experiments`: everything related to the experiments from our paper
  
  - `camera_paths`: camera paths for rendering videos
  
  - `delftblue_docs` DelftBlue Bash code with documentation (incomplete)
  
  - `experiments\scripts` Bash scripts that we used for running the experiments

- `papers`: submodules we used in our work:
  
  - Ha-NeRF — Ha-NeRF (Hallucinated Neural Radiance Fields in the Wild) using pytorch.
  
  - LLFF — Local Light Field Fusion.
  
  - instant-ngp — Instant neural graphics primitives. <u>Our fork</u> adds: a script to convert camera poses from nuScenes to NeRF format (transforms.json); the flag --visualize_cameras to the testbed, useful to debug camera positions.
  
  - nerf_pl — NeRF (Neural Radiance Fields) and NeRF in the Wild using pytorch-lightning.
  
  - nerfstudio — A collaboration friendly studio for NeRFs.
  
  - torch-ngp — A pytorch CUDA extension implementation of instant-ngp (sdf and nerf), with a GUI.

- `scripts`: Python scripts for (i.e.) converting from slices to different input files and generating masks using semantic segmentation

## Project outline

The project consists of three parts:

1) Initial literature study inventorying the current state of NeRF applied to driving scenes
2) An evaluation of current technologies on driving scenes and common failure modes
3) An attempt at improving rendering by removing dynamic objects from the scene

<!--- ## Evaluation of current technologies

### COLMAP

**TODO**

### C2W matrices and inputs

**TODO**

### NDC and scaling

**TODO**
--->

### Results

To consult the results, please refer to our [paper](paper.pdf).

## Usage

#### nuScenes camera poses to NeRF format

Generate transforms.json and hard-link images to the local folder.

PATH_TO_INSTANT_NGP refers to the path where `papers/instant-ngp` was cloned.

```bash
cd $DATASET_DIR
python $PATH_TO_INSTANT_NGP/scripts/dataset2nerf_nuscenes.py --aabb_scale=4 --num_dataset_samples=20 --scene_num 6 --nuscenes_dataroot $NUSCENES_DATAROOT --adaptive_rescale
```

Generate train/test splits

```bash
cd IAAIP-NeRF/scripts/
python split_train_val.py ./split_test/nuscenes6_6cam/ --percent_train 80
```

#### Semantic segmentation for dynamic objects removal

We use [Detectron2]() to generate the segmentation masks.
Follow the instructions in [scripts/mask_generation/README.md](scripts/mask_generation/README.md) to reproduce.