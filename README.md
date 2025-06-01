
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

# Dataset
We used the NVFi(https://github.com/vLAR-group/NVFi) and Dynamic 3D Gaussians datasets (https://github.com/JonathonLuiten/Dynamic3DGaussians).

