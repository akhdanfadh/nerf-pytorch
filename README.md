# radiance-fields PyTorch

A PyTorch *re*-implementation of the seminal [Neural Radiance Field](https://arxiv.org/abs/2003.08934) (NeRF) paper by Mildenhall et al. (2020), a massive milestone in image-based, neural rendering literature.

This repository is under development and will be updated to include more variants and improvement of NeRFs. The goal is to provide a clean and easy-to-understand codebase for practitioners to experiment with NeRF and its variants.

<p align="center">
  <img src="https://datagen.tech/wp-content/uploads/2022/03/image1.png" width=75%> <br>
  An overview of NeRF scene representation and differentiable rendering procedure.
</p>

NeRF itself is a simple fully connected network trained to reproduce input views of a single scene using a rendering loss. The network directly maps from spatial location and viewing direction (5D input), into color and opacity (4D output), acting as the "volume" so we can use volume rendering to differentiably render new image from unseen viewing direction.


## Getting Started

### Installation

We recommend using a virtual environment to install the required packages, such as `conda`.
```bash
git clone git@github.com:akhdanfadh/nerf-pytorch.git
cd nerf-pytorch

conda create -n radiance-fields python=3.10
conda activate radiance-fields

# adjust based on your CUDA environment
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e .  # install the project (and dependencies)
```
This project has been tested on WSL Ubuntu 22.04 with PyTorch 2.2 & CUDA 12.2 on a 3090.

### Usage

Before running our script, register the project root directory as an environment variable to help the Python interpreter search our files. Set this **every time you open a new terminal** to activate the virtual environment.
```bash
export PYTHONPATH=.  # assuming you are in the project root directory already
```

By default, the configuration is set for the `lego` scene included in the `Blender` (nerf-synthetic) dataset. Refer to the config files under `configs` for more details. Executing the following initiates training:
```bash
python runners/train.py
```

All by-products produced during each run, including TensorBoard logs, will be saved under an experiment directory under `outputs`. This is automatically done by [Hydra](https://hydra.cc), the library we use for managing our config files.

There are two framework to monitor training process: [TensorBoard](https://www.tensorflow.org/tensorboard) and [WandB](https://wandb.ai/site), with TensorBoard being the default. If you like to use WandB, set the relevant `use_wandb` parameter in the default config file, and do not forget to set up an API key on your local environment.


## License

This project is licensed under the [MIT License](LICENSE).


## Acknowledgements

Code structure and training loop are based on the [torch-NeRF](https://github.com/DveloperY0115/torch-NeRF) repository with lots of adjustments, typehinting, and comments to make it more understandable.
I would also like to acknowledge [nerf-pytorch](https://github.com/krrish94/nerf-pytorch) repository for further inspiration on how to make the code more efficient.


## Citation

Kudos to the authors of the seminal paper for their amazing work. If you find this code useful, please consider citing the original work:
```
@inproceedings{mildenhall2020nerf,
    title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
    author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
    year={2020},
    booktitle={ECCV},
}
```