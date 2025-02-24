# Reproducible Deep Learning Research with GPU-Jupyter: A Demo-Project

This is a repository to demonstrate how to use **GPU-jupyter for reproducible deep learning research**.

```
Reproducibility is acknowledged as a fundamental requirement of the scientific method.
Yet, many meta-studies highlight a reproducibility crisis across diverse scientific domains.
The rise of machine learning further complicates the replication of research by introducing additional requirements in computational environments.

In this work, we propose GPU-Jupyter, a framework for reproducible research for deep learning.
The experiment setup is encapsulated into an isolated container environment with GPU support and a rich data science toolstack, thus mitigating version conflicts and providing well-defined setup.

We demonstrate three scenarios in which GPU-Jupyter facilitates reproducibility and collaboration among researchers.
Researchers can share their research including the experimental setup in version-controlled and taggable images.
Conversely, existing work, whether published with or without GPU-Jupyter, can be reproduced rapidly and without affecting the existing local setup.

Finally, we compare GPU-Jupyter with alternative solutions, highlighting its ability to meet key requirements for reproducible deep learning research.
By streamlining environment management, GPU-Jupyter lowers the barriers to reproducibility, fostering a more seamless, open, and trustworthy research culture.
```

The latest **GPU-Jupyter** release is built on **CUDA 12.5**, ensuring full compatibility with both **PyTorch** and **TensorFlow** for seamless deep learning experimentation.


## Requirements

To facilitate all the advantages, two requirements are needed on the host system.

### 1. NVIDIA CUDA

NVIDIA is a leading provider of GPU hardware acceleration for deep learning. Therefore,

 powered by its proprietary **CUDA** drivers [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit).

### 2. Containerization for Reproducibility

To enhance reproducibility, GPU-Jupyter leverages **containerization technologies** such as [Docker](https://docker.com/), [Rancher Desktop](https://rancherdesktop.io/), and [Podman](https://podman.io/), in combination with the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit). As illustrated in the project overview, this setup enables containerized applications to directly utilize **NVIDIA GPUs**, ensuring high-performance execution while maintaining an isolated and replicable research environment.

GPU Support is achieved by means of building upon the officially supported NVIDIA CUDA Docker images.
It is worth mentioning that passing the GPU through to containers requires the NVIDIA Container Toolkit on the host system.

NVIDIA GPU,
NVIDIA CUDA drivers
Docker Engine
NVIDIA Container Toolkit


## Reproduce Setup

### Variant 1: Within a Standard GPU-Juypter Image

The project is easy to make reproducible, but more lines of code is required.

```bash
cd path/to/project
docker run --gpus all -it -p 8888:8888  -v $(pwd):/home/jovyan/work -e GRANT_SUDO=yes -e JUPYTER_ENABLE_LAB=yes --user root cschranz/gpu-jupyter:v1.8_cuda-12.5_ubuntu-22.04
```

Open the built-in JupyterLab interface on [http://localhost:8888](http://localhost:8888) and input the access token which is provided in the docker output, for example:

```bash
[C 2025-02-17 12:25:57.988 ServerApp]

    To access the server, open this file in a browser:
        file:///home/jovyan/.local/share/jupyter/runtime/jpserver-22-open.html
    Or copy and paste one of these URLs:
        http://6680bd8aaa39:8888/lab?token=ba872bc692eb7d749bbfaf7ef1a48ce5a8ff3658f2d49b14
        http://127.0.0.1:8888/lab?token=ba872bc692eb7d749bbfaf7ef1a48ce5a8ff3658f2d49b14
```

The access token is `ba872bc692eb7d749bbfaf7ef1a48ce5a8ff3658f2d49b14`. Note the port outputted here can be adapted according to the `docker run` command.

Then open in JupyterLab a new Terminal and run:

```bash
cd work
git clone https://github.com/iot-salzburg/reproducible-research-with-gpu-jupyter.git

cd reproducible-research-with-gpu-jupyter/
pip install -r requirements.txt
```

With `cd work` it is navigated to the work directory of the container, that is mounted on the current working directory on the host server using `-v $(pwd):/home/jovyan/work` of the  `docker run` command.
The setup of the deep learning experiment is reproduced.


### Variant 1: In the Containerized Experiment

The project is containerized on top of the standard GPU-Jupyter image, therefore the whole setup of the whole deep learning experiment can be reproduced in a single line of code:

```bash
docker run ... cschranz/reproducible-research-
```


## Reproduction of the experiment

Then navigate in JuperLab's file explorer into `work/reproducible-research-with-gpu-jupyter/` and check out the project directory.

Run the deployed model with

```bash
python src/deployment/inference.py
```

Train the model in the provided Jupyter Notebook under `src/modelling/train_ResNet.ipynb`.

### More on JupyterLab

TODO: link YouTube Tutorial.


## Containerize your own research

You can do this for both variants, depending on your preferences.


### Configuration of the container for adapting to your project

Use the flag `-d` for detached mode
Set static access token that is not renewed at any new start
Sudo-rights
User root
Try if JUPYTER_ENABLE_LAB is required.

In case there are problems with the accessibility, try running `sudo chown -R jovyan.users work/`
or `sudo chown -R dev.hma Reproducible-Research-with-GPU-Jupyter/` on the host server.

### Cite as

```
Schranz, C., Pilosov, M., Beeking, M. (2025). GPU-Jupyter: A Framework for Reproducible Deep Learning Research.  [Manuscript submitted for publication] In Interdisciplinary Data Science Conference
```


## TODOs

In publication:
- Expose port in docker run (without explicit port mapping it is not exposed)
- Mention the setup can be reproduced in one line of code

- Upload to Arxiv or https://paperswithcode.com/search?q_meta=&q_type=&q=reproducible
 