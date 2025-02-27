# Reproducible Deep Learning Research with GPU-Jupyter: A Demo Project  

This repository demonstrates how to use [**GPU-Jupyter**](https://github.com/iot-salzburg/gpu-jupyter) to ensure **reproducible deep learning research** with minimal setup effort.  

**Reproducibility is a fundamental requirement of the scientific method.** However, many meta-studies highlight a reproducibility crisis across scientific disciplines.

[**GPU-Jupyter**](https://github.com/iot-salzburg/gpu-jupyter) is a flexible and efficient **framework for reproducible deep learning experiments** by encapsulating the entire experimental setup into an isolated, GPU-supported containerized environment. This approach mitigates version conflicts, ensures a well-defined setup, and streamlines the sharing and collaboration of customized images.  

This repository demonstrates:  

- How to **reproduce existing research**.  
- How to **make your own deep learning experiments reproducible** in a single command.  


![Four pillars of reproducibility](data/reproducible_research_low.png)  

Reproducibility in deep learning is facilitated by four key pillars:  

1. **Open Code**: Code and workflows should be shared in a version-controlled manner.  
2. **Open Data**: Publicly available datasets ensure accessibility and verification.  
3. **Controlled Randomness**: Using fixed random seeds ensures deterministic results.  
4. **Reproducible Setup**: Computational environments should be well-defined and portable.  

**GPU-Jupyter specifically addresses the fourth pillar** by a **containerized deep learning setup** that ensures experiments can be executed and verified seamlessly across different systems.  


## Table of Contents

1. [Requirements](#requirements)
2. [Reproduce a Deep Learning Experiment](#reproduce-a-deep-learning-experiment)
   - [Variant 1: Reproducing Existing Work, Published with or without GPU-Jupyter](#variant-1-reproducing-existing-work-published-with-or-without-gpu-jupyter)
   - [Variant 2: Reproducing Existing Work with a Customized GPU-Jupyter Setup](#variant-2-reproducing-existing-work-with-a-customized-gpu-jupyter-setup)
3. [Make Your Own Research Reproducible](#make-your-own-research-reproducible)
   - [Variant 1: Publishing Work in a Standard GPU-Jupyter Environment](#variant-1-publishing-work-in-a-standard-gpu-jupyter-environment)
   - [Variant 2: Publishing in a Customized GPU-Jupyter Image](#variant-2-publishing-in-a-customized-gpu-jupyter-image)
4. [More Information](#more-information)
5. [Cite This Work](#cite-this-work)



---

## Requirements  

To use GPU-Jupyter, the following components must be installed on the host system:

- **NVIDIA GPU**: A compatible NVIDIA GPU is required to accelerate deep learning computations, leveraging its parallel processing capabilities for efficient matrix operations.  
- **NVIDIA CUDA Drivers**: The CUDA toolkit provides the necessary drivers and runtime libraries that enable GPU acceleration for deep learning frameworks like PyTorch and TensorFlow. GPU-Jupyter is built on NVIDIA’s official CUDA Docker images to ensure compatibility. Install the latest version from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).  
- **Docker Engine**: [Docker](https://www.docker.com/get-started) ensures a reproducible execution environment by isolating deep learning experiments from the host system, preventing dependency conflicts and facilitating portability.  
- **NVIDIA Container Toolkit**: This toolkit allows Docker containers to access GPU resources on the host system, ensuring full hardware acceleration inside the containerized deep learning environment. Official installation instructions are available on [NVIDIA’s GitHub repository](https://github.com/NVIDIA/nvidia-container-toolkit).  


---



## Reproduce a Deep Learning Experiment

### Variant 1: Reproducing Existing Work, Published with or without GPU-Jupyter

To reproduce a deep learning experiment, follow these steps:  

**1. Start a GPU-Jupyter container**  

Run the following command, replacing `$(pwd)` with the absolute path to your project directory:  

```bash
cd path/to/project
docker run --gpus all -it -p 8888:8888 \
    -v $(pwd):/home/jovyan/work \
    -e GRANT_SUDO=yes -e JUPYTER_ENABLE_LAB=yes \
    -e NB_UID=$(id -u) -e NB_GID=$(id -g) \
    --user root cschranz/gpu-jupyter:v1.8_cuda-12.5_ubuntu-22.04
```

Note the following Docker parameters:

- **`--gpus all`**: Grants the container access to all available GPUs on the host system, enabling GPU acceleration for deep learning workloads.  
- **`-it`**: Runs the container in **interactive mode**, allowing direct user interaction via a terminal (useful for executing commands inside the container).  
- **`-p 8888:8888`**: Maps **port 8888** of the container to **port 8888** on the host, enabling access to JupyterLab through `http://localhost:8888`.  
- **`-v $(pwd):/home/jovyan/work`**: Mounts the current directory (`$(pwd)`) on the host to `/home/jovyan/work` inside the container, ensuring persistent access to files and code across sessions.  
- **`-e GRANT_SUDO=yes`**: Grants the Jupyter user (`jovyan`) sudo privileges inside the container, allowing administrative commands if needed.  
- **`-e JUPYTER_ENABLE_LAB=yes`**: Ensures JupyterLab (instead of the classic Jupyter Notebook interface) is enabled when the container starts.  
- **`-e NB_UID=$(id -u) -e NB_GID=$(id -g)`**: Sets the **user ID (UID) and group ID (GID)** inside the container to match the host system’s user, preventing permission issues when accessing mounted files.  
- **`--user root`**: Runs the container as the **root user**, allowing unrestricted access to system configurations and software installations within the container.  




**2. Open JupyterLab**

Access JupyterLab at [http://localhost:8888](http://localhost:8888) and enter the access token printed in the Docker output:

```bash
[C 2025-02-17 12:25:57.988 ServerApp]

    To access the server, open this file in a browser:
        file:///home/jovyan/.local/share/jupyter/runtime/jpserver-22-open.html
    Or copy and paste one of these URLs:
        http://127.0.0.1:8888/lab?token=ba872bc692eb7d749bbfaf7ef1a48ce5a8ff3658f2d49b14
```

Then navigate in JuperLab's file explorer into `work/reproducible-research-with-gpu-jupyter/` and check out the project directory.


**3. Clone the Experiment Repository**

Inside JupyterLab, open a terminal and execute:

```bash
cd work
git clone https://github.com/iot-salzburg/reproducible-research-with-gpu-jupyter
cd reproducible-research-with-gpu-jupyter/
pip install -r requirements.txt
```

This **recreates the original deep learning environment**, ensuring that all dependencies match the original research setup.

Then, reproduce the experiment by running the provided Jupyter Notebook under `src/modelling/train_ResNet.ipynb`.
Randomness is controlled by providing random seeds for functions in numpy, PyTorch, and PyTorch-CUDA.


#### Alternative Setup Using Docker Compose for Configuration Management

Instead of specifying all parameters manually in a `docker run` command, **Docker Compose** allows for a structured, version-controlled setup by defining configurations in the local `docker-compose.yml` file. This approach is especially useful for managing multiple services, maintaining reproducibility, and simplifying container deployment.  

Using a `docker-compose.yml` has benefits in terms of:

- **Version-controlled configuration**: Easily track changes in Git.  
- **Reusable setup**: Simplifies launching the environment across different systems.  
- **Automatic container management**: Easily start, stop, and restart the service.  

**Basic `docker-compose` Commands**  

1. **Start GPU-Jupyter** (in the background):  
   ```bash
   docker-compose up -d
   ```
   - Builds (if necessary) and starts the container based on `docker-compose.yml`.  
   - The `-d` flag runs it in **detached mode**, meaning the container continues running in the background.  

2. **Stop and remove the container**:  
   ```bash
   docker-compose down
   ```
   - Stops and removes the running container(s) and associated network.  

3. **View logs in real time**:  
   ```bash
   docker-compose logs -f
   ```
   - Displays container logs, useful for debugging and monitoring.  




### Variant 2: Reproducing Existing Work with a Customized GPU-Jupyter Setup  

For fully automated reproducibility, the entire experiment—including **code, dependencies, and configurations**—can be packaged into a pre-configured **Docker image**. This approach eliminates manual setup steps, ensuring a seamless reproduction process.  

If the authors have **built and published a customized GPU-Jupyter image** to a container registry (e.g., DockerHub), the entire deep learning experiment can be **reproduced with a single command**:  

```bash
docker run --gpus all --rm -it -p 8888:8888 cschranz/reproducible-research-with-gpu-jupyter:v1.0
```

**Key Considerations:**  

- **Use the exact image provided by the experiment's authors** to ensure a **fully consistent computational environment**.  
- The example image `cschranz/reproducible-research-with-gpu-jupyter:v1.0` demonstrates a reproducible setup, but this should be replaced with the actual image reference from the research publication.  
- **No additional dependencies need to be installed**—the environment is already pre-configured within the container.  

This approach ensures that **any researcher can replicate the original computational setup** with minimal effort, making deep learning research more accessible, reliable, and reproducible.





## Make Your Own Research Reproducible  

GPU-Jupyter enables researchers to publish experiments with **full computational reproducibility**.  

### Variant 1: Publishing Work in a Standard GPU-Jupyter Environment

1. Develop deep learning models in **JupyterLab** inside a **GPU-Jupyter** container.  
2. Share the **code and dataset** in a public Git repository and/or data repository.  
3. Specify the **exact GPU-Jupyter image** used in the research paper.  
4. Explain all additional steps required for the setup and reproduction of the experiment.

Example Open Science statement for a publication:  

> The authors ensure reproducibility by providing all code, data, and environment details at  
> [https://github.com/iot-salzburg/reproducible-research-with-gpu-jupyter](https://github.com/iot-salzburg/reproducible-research-with-gpu-jupyter).  
> All experiments were conducted using the **GPU-Jupyter** image:  
> `cschranz/gpu-jupyter:v1.8_cuda-12.5_ubuntu-22.04`  


### Variant 2: Publishing in a Customized GPU-Jupyter image 


For seamless **reproducibility in one single command**, publish a **customized GPU-Jupyter image** that includes the entire experiment.  

**Step 1: Define the Dockerfile**

Build a **Dockerfile** that installs all required dependencies. Use an appropriate tagged GPU-Jupyter image as base image and declare  the full computational setup of your experiment from there, such as installations in package managers like `pip`. Use the local `Dockerfile` as an example.

**Step 2: Log in to Docker Hub**

```sh
docker login
```
- Enter your **Docker Hub username** and **password** when prompted.
- If using **Access Tokens**, generate one from [Docker Hub](https://hub.docker.com/settings/security) and use it as the password.


**Step 3: Build the Docker Image**

```sh
docker build -t your-dockerhub-username/image-name:tag .
```

- `-t` specifies the **tag**. Specify your own username and repository name for `your-dockerhub-username/image-name:tag`.
- `.` means the **Dockerfile is in the current directory**.

Verify the image is built using:

```sh
docker images
```


**Step 4: Push the Image to DockerHub**

```sh
docker push your-dockerhub-username/image-name:tag
```

- The image will now be available on **Docker Hub** at:
  ```
  https://hub.docker.com/r/your-dockerhub-username/image-name
  ```



**Step 5 (optional): Verify if it worked: Pull and Run the Image**

To test on another machine, pull and run your image using:

```sh
docker run --gpus all --rm -it -p 8888:8888 your-dockerhub-username/image-name:tag
```

To reproduce the experiment within the repository, run:

```sh
docker run --gpus all --rm -it -p 8888:8888 cschranz/reproducible-research-with-gpu-jupyter:v1.0
```


Include the **single-command execution** for reproducing your deep learning experiment in the research paper.
This ensures that **future researchers can reproduce your work effortlessly** without additional setup steps.


Make sure that the data within the repository is reasonable sized or downloaded during execution of the experiment. Otherwise, a big dataset might be included in the image stored on DockerHub which is a redundant storage of data.

<!-- 
TODO: commit and push the image as it is

# 1️⃣ Find the running container
docker ps

# 2️⃣ Commit the current state of the container to a new image
docker commit <container_id> your-dockerhub-username/image-name:tag

# Example:
docker commit 2af096ff9e32 cschranz/myproject:updated

# 3️⃣ Verify the new image exists
docker images

# 4️⃣ Log in to Docker Hub (if not already logged in)
docker login

# 5️⃣ Push the new image to Docker Hub
docker push your-dockerhub-username/image-name:tag

# Example:
docker push cschranz/myproject:updated

# 6️⃣ (Optional) Tag the new image as "latest" and push it
docker tag your-dockerhub-username/image-name:tag your-dockerhub-username/image-name:latest
docker push your-dockerhub-username/image-name:latest

# Example:
docker tag cschranz/myproject:updated cschranz/myproject:latest
docker push cschranz/myproject:latest

# 7️⃣ (Optional) Pull the updated image on another machine
docker pull your-dockerhub-username/image-name:tag

# Example:
docker pull cschranz/myproject:updated

# 8️⃣ (Optional) Run the new image
docker run -it --rm your-dockerhub-username/image-name:tag

# Example:
docker run -it --rm cschranz/myproject:updated
-->


## More information

### Learn More About JupyterLab  

<!-- 
TODO: link YouTube Tutorial.
-->
Here is a great tutorial on how to use JupyterLab:

[![Project Jupyter: How to Use JupyterLab](https://i3.ytimg.com/vi/A5YyoCKxEOU/maxresdefault.jpg)](https://www.youtube.com/watch?v=A5YyoCKxEOU)


### Customizing the Docker Container for Your Project  

GPU-Jupyter allows flexible configuration for **various deep learning workflows**:  

- Run **detached containers** using `-d` to keep them running in the background.  
- Set a **static access token** for persistent authentication as described among other configurations in [github.com/GPU-Jupyter](https://github.com/iot-salzburg/gpu-jupyter).  
- Enable **sudo privileges** within the container using `-e GRANT_SUDO=yes`.  

For permission issues, ensure proper ownership:  

```bash
sudo chown -R jovyan.users work/
```


## Cite This Work  

When you are using GPU-Jupyter for the development of your academic work and its reproduction, please cite the framework in your publication as:  

```
Schranz, C., Pilosov, M., Beeking, M. (2025).  
GPU-Jupyter: A Framework for Reproducible Deep Learning Research.  
[Manuscript submitted for publication] In Interdisciplinary Data Science Conference.  
```

--- 

By streamlining environment management, **GPU-Jupyter reduces barriers to reproducibility**, fostering an open and trustworthy research culture. 



