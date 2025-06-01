
# ParticleGS: Particle-Based Dynamics Modeling of 3D Gaussians for Prior-free Motion Extrapolation



Our core idea is to emulate existing classical particle dynamics systems by introducing a latent vector that implicitly represents the dynamics state of Gaussian particles, thereby enabling extrapolation.


# Demo
More demos can be found in ./demos .
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
python train.py -s ./data/NVFi_datasets/InDoorObj/data/telescope -m ./output/telescope --conf ./arguments/nvfiobj/telescope.py
# render
python render.py --conf ./arguments/nvfiobj/telescope.py -m ./output/telescope --iteration best
```

# Dataset
We used the NVFi and Dynamic 3D Gaussians datasets. Datasets can be organized as follows:

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