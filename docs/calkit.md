# Turning this into a Calkit project

## Setup

Create a virtual-env:

```bash
cd reproducible-research-with-gpu-jupyter

virtualenv .venv
source .venv/bin/activate
```

Install calkit via pip

```bash
python -m pip install calkit-python
```



## Create the project from the GitHub repo

This was done in a fork, so keep in mind `calkit config remote` will
need to be run in the main project if the PR is merged
and the DVC remote (which could be used for model storage) is needed/desired.
The `iot-salzburg` org and project will also need to be created on calkit.io.

```sh
calkit new project . \
  --cloud \
  --public \
  --title \
    "Reproducible Deep Learning Research with GPU-Jupyter: A Demo Project" \
  --description \
    "A fork of iot-salzburg/reproducible-research-with-gpu-jupyter."
```

## Create the Docker environment for the project

In this case, we'll call it `main`, but we can call it whatever we like:

```sh
calkit new docker-env \
  --name main \
  --path Dockerfile \
  --dep requirements.txt \
  --env-var GRANT_SUDO=yes \
  --env-var JUPYTER_ENABLE_LAB=yes \
  --env-var NB_UID='$(id -u)' \
  --env-var NB_GID='$(id -g)' \
  --user root \
  --gpus all \
  --platform linux/amd64 \
  --port 8888:8888 \
  --wdir /home/jovyan/work
```

## Create the Jupyter notebook pipeline stage

We declare that this notebook should be run in the `main` environment:

```sh
calkit new jupyter-notebook-stage \
  --name run-notebook \
  --environment main \
  --notebook-path src/Finetune_ResNet.ipynb \
  --out-git src/model.pt \
  --out-git src/ResNet_results.csv \
  --html-storage git \
  --cleaned-ipynb-storage git \
  --executed-ipynb-storage git
```

## Declare the notebook

This makes it possible to view rendered HTML (including interactive elements)
on calkit.io.

```sh
calkit new notebook \
  src/Finetune_ResNet.ipynb \
  --title \
    "Reproducible Deep Learning Experiment: Land Use Classification with EuroSAT" \
  --description "The main training and evaluation notebook." \
  --stage run-notebook
```

## Now run the pipeline and save the project

```sh
calkit run
```

If everything ran okay, we can commit and back up to the cloud with:

```sh
calkit save -am "Run pipeline"
```

## Working interactively

If we want to run the notebook interactively, e.g., to generate some
different plots,
we can spin up JupyterLab in our `main` Docker environment with:

```sh
calkit xenv -n main jupyter lab
```

After we're happy with the results, the pipeline should be rerun
with `calkit run`.
