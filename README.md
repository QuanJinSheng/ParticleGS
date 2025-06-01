
# ParticleGS: Particle-Based Dynamics Modeling of 3D Gaussians for Prior-free Motion Extrapolation

# Demo

![](./demos/chessboard_demo.gif)

# Install

```shell
git clone https://github.com/QuanJinSheng/ParticleGS.git
cd ParticleGS

conda env create -f environment.yaml
conda activate ParticleGS

# require CUDA 11.8
pip install -e ./submodules/diff-gaussian-rasterization
pip install -e ./submodules/simple-knn
```
# Run
Our project structure is similar to the standard 3DGS.
```shell

# train
python train.py -s ./data/NVFi_datasets/InDoorObj/data/telescope -m ./output/telescope --conf ./arguments/nvfiobj/telescope
# render
python render.py --conf ./arguments/nvfiobj/telescope -m ./output/telescope --iteration best
```
# Dataset
We used the [NVFi](https://github.com/vLAR-group/NVFi) and [Dynamic 3D Gaussians](https://github.com/JonathonLuiten/Dynamic3DGaussians) datasets. Datasets can be organized as follows:

```shell
data
├── NVFi_datasets
│   ├── InDoorObj/
│   │   ├── telescope
│   │   ├── bat
│   │   └── ...
│   ├── InDoorSeg
│   │   ├── chessboard
│   │   └── ...
├── PanopticSports/
│   ├── boxes
│   └── ...

```

# Acknowledgments
We sincerely thank the authors of [NVFi](https://github.com/vLAR-group/NVFi), [Dynamic 3D Gaussians](https://github.com/JonathonLuiten/Dynamic3DGaussians), and [Grid4D](https://github.com/JiaweiXu8/Grid4D.git), whose codes and datasets were used in our work.
