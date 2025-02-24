# Reproducible Deep Learning Research with GPU-Jupyter: A Demo-Project

This is a repository to demonstrate how to use **GPU-jupyter for reproducible deep learning research** in only **one single command**.

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

![Four pillars of reproducibility](data/reproducible_research_low.png)
The four pillars of reproducibility.

This project demonstrates how deep learning experiments can be made reproducible, by following each of the four pillars for reproducible research.
In particular, GPU-Jupyter facilitates to create a reproducible setup in only one line of code.


**GPU-Jupyter** enables high-performance execution while maintaining an isolated and replicable research environment.
GPU Support is achieved by means of building upon the officially supported NVIDIA CUDA Docker images.
The latest **GPU-Jupyter** release is built on **CUDA 12.5**, ensuring full compatibility with both **PyTorch** and **TensorFlow** for seamless deep learning experimentation.


## Requirements

To facilitate that the Docker containers have a GPU-support, four requirements are needed on the host system.

- **NVIDIA GPU**: A physical graphics card (GPU) is required on the host server
- **NVIDIA CUDA drivers**: NVIDIA is a leading provider of GPU hardware acceleration for deep learning. Therefore, the image is based on the officially supported **NVIDIA CUDA** drivers [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit).
- **Docker Engine**: To enable reproducibility, GPU-Jupyter utilizes the **containerization technology** [Docker](https://docker.com/). Make sure Docker Engine is installed on the host server.
- **NVIDIA Container Toolkit**: Install the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) in order to make the GPU accessible within Docker containers.



## Reproduce the Experiment's Setup



### Option 1: The Research is available as tagged image

The project is easy to make reproducible, but more lines of code is required.

```bash
cd path/to/project
docker run --gpus all -it -p 8888:8888 -v $(pwd):/home/jovyan/work -e GRANT_SUDO=yes -e JUPYTER_ENABLE_LAB=yes -e NB_UID=$(id -u) -e NB_GID=$(id -g) --user root cschranz/gpu-jupyter:v1.8_cuda-12.5_ubuntu-22.04
```

Summary of the parameters:

- `--gpus all`: Make all detected GPUs of the host server accessible within the container.
- `-it`
- `-p 8888:8888`: Forward Jupyter on port 8888
- `-v $(pwd):/home/jovyan/work` mount directory
- `-e GRANT_SUDO=yes`
- `-e JUPYTER_ENABLE_LAB=yes`
- `-e NB_UID=$(id -u) -e NB_GID=$(id -g)`
- `--user root`
- `cschranz/gpu-jupyter:v1.8_cuda-12.5_ubuntu-22.04`: run a container based on this tagged GPU-Jupyter image


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


Then navigate in JuperLab's file explorer into `work/reproducible-research-with-gpu-jupyter/` and check out the project directory.
Train the model in the provided Jupyter Notebook under `src/Finetune_ResNet.ipynb`.


### Option 2: The Research is NOT available as tagged Image - Use a Standard GPU-Juypter Image

The project is easy to make reproducible, but more lines of code is required.

```bash
cd path/to/project
docker run --gpus all -it -p 8888:8888 -v $(pwd):/home/jovyan/work -e GRANT_SUDO=yes -e JUPYTER_ENABLE_LAB=yes -e NB_UID=$(id -u) -e NB_GID=$(id -g) --user root cschranz/gpu-jupyter:v1.8_cuda-12.5_ubuntu-22.04
```

Then open in JupyterLab a new Terminal and run:

```bash
cd work
git clone https://github.com/iot-salzburg/reproducible-research-with-gpu-jupyter

cd reproducible-research-with-gpu-jupyter/
pip install -r requirements.txt
```

With `cd work` it is navigated to the work directory of the container, that is mounted on the current working directory on the host server using `-v $(pwd):/home/jovyan/work` of the  `docker run` command.
The setup of the deep learning experiment is reproduced.






## Make your research reproducible in one single Command

The project is containerized on top of the standard GPU-Jupyter image, therefore the whole setup of the whole deep learning experiment can be reproduced in a single line of code:
You can do this for both options, depending on your preferences.


### Option 1: Containerize the Experiment in a Dockerfile

In this option, a dedicated Dockerfile is created that uses GPU-Jupyter as base image. In the example in `Dockerfile`, (1) `apt-utils` are installed, (2) the `requirements.txt`-file is copied and installed via pip, (3) the code as well as the data is copied, and (4) the user permissions are adapted such that no files or folders are hidden in JupyterLab.

In many cases, it is recommended that the data is not copied into the container, but externally referenced.

```bash
docker build -t gpu-jupyter_repro .
docker run --rm --gpus all -d -it -p 8848:8888 -e GRANT_SUDO=yes -e JUPYTER_ENABLE_LAB=yes --user root --name gpu-jupyter_repro_1 gpu-jupyter_repro
docker exec -it gpu-jupyter_repro_1 jupyter server list  # output token
```

Then push the image to DockerHub using your DockerHub's organization or username as namespace.

```bash
docker login
docker build -t [YOUR_USERNAME]/reproducible-research-with-gpu-jupyter:v1.0 .
docker push [YOUR_USERNAME]/reproducible-research-with-gpu-jupyter:v1.0
```



### Option 3: Push the Container as Image as it is

While this option is faster, make sure that the setup on top of the basic GPU-Jupyter image requires minimal lines of installations.


```bash
# Find the running container
docker ps

# Commit the current state of the container to a new image
docker commit [CONTAINER_ID] [YOUR_USERNAME]/reproducible-research-with-gpu-jupyter:v1.0

# Verify the new image exists (grep filters lines by a keyword)
docker images | grep reproducible

# Log in to Docker Hub (if not already logged in)
docker login

# Push the new image to Docker Hub
docker push [YOUR_USERNAME]/reproducible-research-with-gpu-jupyter:v1.0
```






## More information

### More on JupyterLab

TODO: link YouTube Tutorial.

### Configuration of the container for adapting to your project

Use the flag `-d` for detached mode.

Set static access token that is not renewed at any new start
Sudo-rights
User root
Try if JUPYTER_ENABLE_LAB is required.

In case there are problems with the accessibility, try running `sudo chown -R jovyan.users work/` within the container run with root priviledges.


### Cite as

If you are using the GPU-Jupyter framework or one of its images, please refer to:

```
Schranz, C., Pilosov, M., Beeking, M. (2025). GPU-Jupyter: A Framework for Reproducible Deep Learning Research.  [Manuscript submitted for publication] In Interdisciplinary Data Science Conference
```


## TODOs

Double-check permissions:

```bash
sudo chown -R dev.hma reproducible-research-with-gpu-jupyter/
sudo chmod -R g+X reproducible-research-with-gpu-jupyter/
sudo ls -lah reproducible-research-with-gpu-jupyter/
cd reproducible-research-with-gpu-jupyter/
git config --global --add safe.directory /home/hma/MDIlab/gpu_jupyter_project/Reproducible-Research-with-GPU-Jupyter/reproducible-research-with-gpu-jupyter
```

In publication:
- Expose port in docker run (without explicit port mapping it is not exposed)
- Mention the setup can be reproduced in one line of code

- Upload to Arxiv or https://paperswithcode.com/search?q_meta=&q_type=&q=reproducible
 