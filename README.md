<p align="center">
  <img src="https://d2k0ddhflgrk1i.cloudfront.net/Websections/Huisstijl/Bouwstenen/Logo/02-Visual-Bouwstenen-Logo-Varianten-v1.png"/><br>
  <a href="">Report</a>
  <br><br>
</p>

# Rendering forward facing driving scenes using Neural Radiance Fields

This repository contains the code for the Interdisciplinary Advanced AI Project (IAAIP) of the TU Delft. During this project we have worked on `Rendering forward facing driving scenes using Neural Radiance Fields`. The project took 10 weeks in total.

The repository structure is as follows:

* `slices`: directory containing slices of different datasets that were used to evaluate different systems.
* `scripts`: directory containing scripts for (i.e.) converting from slices to different input files and generating masks using semantic segmentation.
* `models`: directory containing trained models.

## The project

The project consists of three parts:

1) Initial literature study inventorying the current state of NeRF in rendering forward facing driving scenes
2) An evaluation of current technologies on driving scenes and common failure modes
3) An attempt at improving rendering by removing dynamic objects from the scene

## Evaluation of current technologies

### COLMAP

**TODO**

### C2W matrices and inputs

**TODO**

### NDC and scaling

**TODO**

### Papers

The `papers/`directory contains submodules we used in our work:

- Ha-NeRF &mdash; Ha-NeRF (Hallucinated Neural Radiance Fields in the Wild) using pytorch.

- LLFF &mdash; Local Light Field Fusion.

- instant-ngp &mdash; Instant neural graphics primitives. <u>Our fork</u> adds: a script to convert camera poses from nuScenes to NeRF format (transforms.json); the flag --visualize_cameras to the testbed, useful to debug camera positions. 

- nerf_pl &mdash; NeRF (Neural Radiance Fields) and NeRF in the Wild using pytorch-lightning.

- nerfstudio &mdash; A collaboration friendly studio for NeRFs.

- torch-ngp &mdash; A pytorch CUDA extension implementation of instant-ngp (sdf and nerf), with a GUI.

### Results

To consult the results, please refer to our [paper](paper.pdf).

## Using semantic segmentation for dynamic objects

We use [Detectron2]() to generate the segmentation masks.
Follow the instructions in [scripts/mask_generation/README.md](scripts/mask_generation/README.md) to reproduce.