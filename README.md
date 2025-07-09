# L<sup>3</sup>PS
[**arXiv**](https://arxiv.org/abs/2503.02372) | [**Website**](https://l3ps.cs.uni-freiburg.de/) | [**Video**](https://youtu.be/jfTeCV9Drz0)

This repository is the official implementation of the paper:

> **Label-Efficient LiDAR Panoptic Segmentation**
>
> [Ahmet Selim Çanakçı](https://aselimc.github.io)&ast;, [Niclas Vödisch](https://vniclas.github.io/)&ast;, [Kürsat Petek](http://www2.informatik.uni-freiburg.de/~petek/), [Wolfram Burgard](https://www.utn.de/person/wolfram-burgard/), and [Abhinav Valada](https://rl.uni-freiburg.de/people/valada). <br>
> &ast;Equal contribution. <br> 
> 
> *arXiv preprint arXiv:2503.02372*, 2025.

<p align="center">
  <img src="./assets/l3ps_overview.png" alt="Overview of L3PS approach" width="800" />
</p>

If you find our work useful, please consider citing our paper:
```
@article{canakci2025l3ps,
  author={Canakci, Ahmet Selim and Vödisch, Niclas and Petek, Kürsat and Burgard, Wolfram and Valada, Abhinav},
  title={Label-Efficient LiDAR Panoptic Segmentation},
  journal={arXiv preprint arXiv:2503.02372},
  year={2025},
}
```

**Make sure to also check out our previous works on this topic:**
- [**SPINO**](https://github.com/robot-learning-freiburg/SPINO)
- [**PASTEL**](https://github.com/robot-learning-freiburg/PASTEL)


## 📔 Abstract

A main bottleneck of learning-based models for robotic perception lies in minimizing the reliance on extensive training data while ensuring reliable predictions. In the context of LiDAR panoptic segmentation, this challenge is amplified by the need to handle the dual tasks of semantic and instance segmentation within complex, high-dimensional point cloud data. In this work, we address the problem of Limited-Label LiDAR Panoptic Segmentation (L³PS) by bootstrapping the recent advancements of label-efficient vision panoptic segmentation methods. We propose a technique that leverages a 2D label-efficient network to generate panoptic pseudo-labels from a minimal set of annotated images, which are then projected into point clouds. Utilizing the geometric properties of the point clouds, we refine these pseudo-labels through clustering techniques and enhance their accuracy by accumulating scans over time and separating ground points. The resulting pseudo-labels train an off-the-shelf LiDAR panoptic segmentation network for real-time deployment. Our approach substantially reduces the annotation burden while achieving competitive performance, offering an alternative to previous label-efficient methods.


## 👩‍💻 Code

### 📦 Installation

#### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- CMake 3.10+ (for Patchwork++)
- Eigen3 (for Patchwork++)

#### PASTEL Installation

1. **Create and activate a virtual environment**
   ```bash
   # Using conda (recommended)
   conda create -n pastel python=3.8
   conda activate pastel
   ```

2. **Install PyTorch with CUDA support**
   ```bash
   # Install specific PyTorch versions used by PASTEL
   pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
   ```

3. **Install PASTEL requirements:**
   ```bash
   cd pastel
   pip install -r requirements.txt
   ```

#### L3PS Installation

1. **Create and activate a virtual environment for L3PS**
   ```bash
   # Using conda (recommended)
   conda create -n l3ps python=3.8
   conda activate l3ps
   ```

2. **Install L3PS requirements**
   ```bash
   cd l3ps
   pip install -r requirements.txt
   ```

3. **Install Patchwork++** (You can also install Patchwork++ from the original [repository](https://github.com/url-kaist/patchwork-plusplus))
   ```bash
   # Navigate to the patchwork-plusplus directory
   cd patchwork-plusplus
   
   # Install Python bindings
   make pyinstall
   ```

### 🔧 Model Weights Download

Download the pre-trained PASTEL model weights:

**PASTEL model weights (trained on nuImages):**
- **Semantic head:** [semantic_nuimages.ckpt](https://l3ps.cs.uni-freiburg.de/downloads/semantic_nuimages.ckpt)
- **Boundary head:** [boundary_nuimages.ckpt](https://l3ps.cs.uni-freiburg.de/downloads/boundary_nuimages.ckpt)

Place these weights in the `pastel/` directory or update the config files with the correct paths.

### 🚀 Usage

#### PASTEL: Fine-tuning and Inference

1. **Fine-tune the semantic and boundary heads:**
  ```bash
  cd pastel
  sh finetune_semantic.sh
  sh finetune_boundary.sh
  ```

2. **Run inference:**
  ```bash
  sh inference.sh
  ```

#### L3PS: Creating Pseudo-labeled Point Clouds

1. **Configure the pipeline:**
   Edit `l3ps/configs/config.yaml` to set your dataset paths and parameters:
   ```yaml
   dataset:
     path: /path/to/your/nuscenes/dataset
     version: v1.0-trainval
     split: val
   
   generate:
     pastel_labels_path: workdir/2D_panoptic_labels/${dataset.split}
     output_path: workdir/primal/${dataset.split}
   ```

2. **Run the complete L3PS pipeline:**
   ```bash
   cd l3ps
   python main.py
   ```

   This will execute all four stages:
   - **Generate:** Project 2D PASTEL pseudo-labels to 3D point clouds
   - **Accumulate:** Accumulate multiple scans using KISS-ICP
   - **Refine:** Apply clustering and geometric refinement
   - **Evaluate:** Compute evaluation metrics

3. **Run individual stages:**
   ```bash
   # Generate 3D pseudo-labels from 2D labels
   python generate.py
   
   # Accumulate scans over time
   python accumulate.py
   
   # Refine with clustering and geometric constraints
   python refine.py
   
   # Evaluate the results
   python evaluate.py
   ```

### 📊 Configuration

#### PASTEL Configuration
- **Semantic fine-tuning:** `pastel/configs/semantic_finetune.yaml`
- **Boundary fine-tuning:** `pastel/configs/boundary_finetune.yaml`
- **Inference:** `pastel/configs/panoptic_inference.yaml`

#### L3PS Configuration
- **Main config:** `l3ps/configs/config.yaml`

Update these configuration files according to your dataset paths, hardware setup, and training preferences.

### 📁 Output Structure

After running the L3PS pipeline, the output will be organized as follows:
```
workdir/
├── 2D_panoptic_labels/     # PASTEL pseudo-labels (2D)
├── primal/                 # Projected 3D labels
├── accumulate/             # Accumulated multi-scan labels
├── refine/                 # Refined pseudo-labels
└── logs/                   # Evaluation logs and metrics
```


## 👩‍⚖️  License

For academic usage, the code is released under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license.
For any commercial purpose, please contact the authors.

For [Patchwork++](https://github.com/url-kaist/patchwork-plusplus) and [PASTEL](https://github.com/robot-learning-freiburg/PASTEL), the original licenses apply.


## 🙏 Acknowledgment

We thank the authors of [Patchwork++](https://github.com/url-kaist/patchwork-plusplus) for publicly releasing their [source code](https://github.com/url-kaist/patchwork-plusplus).


This work was funded by the German Research Foundation (DFG) Emmy Noether Program grant number 468878300.
<br><br>
<p float="left">
  <a href="https://www.dfg.de/en/research_funding/programmes/individual/emmy_noether/index.html"><img src="./assets/dfg_logo.png" alt="drawing" height="100"/></a>  
</p>
