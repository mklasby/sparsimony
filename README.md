# sparsimony
![CI Pipeline](https://github.com/mklasby/sparsimony/actions/workflows/ci.yaml/badge.svg)
![CD Pipeline](https://github.com/mklasby/sparsimony/actions/workflows/cd.yaml/badge.svg)

<!-- ## pytorch-poetry-hydra-template
This *opinionated* template is intended to accelerate your deep learning research.
The full stack includes:
* Python >= 3.10
* CUDA >= 12.1
* pytorch >= 2.2.1
* hydra >= 1.3.2
* poetry >= 1.8.1
* wandb >= 0.16.3

## Getting started
Steps to initialize your project:
1. Create your project repo using this template
2. `cd` into the project working directory and run `python ./scripts/init_proj.py`
3. Respond to prompts from CLI. Your responses will be customize the template to your information and project specifics. You will be prompted to provide the following information:
   1. \<\<name\>\>: Name for git config
   2. \<\<email\>\>: Email for git config and/or slurm begin/end/fail notices
   3. \<\<drac-account\>\>: Digital Research Alliance of Canada compute account (def-xxxx). Typically your PI's account. 
   4. \<\<working-dir\>\>: Full path to working directory. Will be used in local as well as docker containers. 
   5. \<\<project-name\>\>: Name of project and python package
   6. \<\<repo-url\>\>: The URL of the project on github (www.github.com/`<user>`/`<repo-name>`)
   7. \<\<wandb-project\>\>: Weights and bias' project name
   8. \<\<wandb-entity\>\>: Weights and bias' entity name
   9. \<\<wandb-api-key\>\>: Weights and bias' API key. NOTE: This information is stored only in `.env` which is not tracked using `git`
4. Create your working environment using `venv` OR `docker`:
   1. `venv`: Simply run the following:
      ```bash
      python3 -m venv .venv
      source .venv/bin/activate
      pip3 install --upgrade pip
      pip3 install poetry==${poetry-version}
      poetry install -vvv
      git submodule init && git submodule update
      ```
   2. `docker`:
      1. vscode:
         1. Populate `customizations.mounts` for any data that you want mounted to dev container
         2. Run `Dev Containers: Reopen in container` in vscode
      2. CLI: run the following:
      ```bash
      docker build --file ./Dockerfile.dev -t ${your-tag-name} --shm-size=16gb .
      docker create --env-file .env --gpus all -i -t --name ${your-container-name} ${your-tag-name}:latest
      docker start -itd --env-file ./.env --mount source=${path-to-scratch},target=\scratch,type=bind \
        --gpus all --shm-size 16G  ${your-container-name}
      ```

## Repository Structure
* `.devcontainer`: Contains .json file for using vscode's devcontainer functionality
* `.github`: CI/CD pipeline definitions
* `.vscode`: vscode settings to play nice with our dev environment including integration with `black` and `flake8`
* `artifacts`: Store your model outputs, logs, and other artifacts generated during training here
* `configs`: hydra configs for reproducible and extensible experimentation. Config groups match structured of python package subdirectories to help users quickly find configs as required. Groups can be extended with multiple configurations and selected using `config.yaml` defaults list. `configs/sweep` is intended for `wandb` sweep configurations and is not used by `hydra`.
* `data`: small datasets such as MNIST and CIFAR-10 can go here. Larger datasets are assumed to be stored outside the project directory such as `/scratch` and mounted to docker containers. 
* `notebooks`: Jupyter notebooks and some suggested style files for nice `matplotlib` plots. Also includes `main.ipynb` which is intended to be synced with `main.py` for easy debugging and hacking of main script. Consider using `jupytext` to sync `.py` files to github rather than `.ipynb`. Note that the `main.ipynb` file can be regenerated anytime by running `./scripts/generate_main_notebook.sh`.
* `scripts`: Various helper scripts for building your project on a new host or on Digital Alliance of Canada nodes. Also includes utilities for downloading `wandb` run tables and `imagenet`. You will also find `init_proj.py` here which is the starter script for initializing the template. 
* `src`: Source files for your project. It is expected that at least one python package will be created in ./src/\<\<package-name\>\>. However, you can have as many packages as you like. Simply add any other packages to `pyproject.toml::packages`
* `tests`: Unit tests using `pytest`. Predefined tags include `slow`, `integration`, and `dist` to help limit your CI/CD to applicable tests.
* `third-party`: Third party dependencies managed with `git submodule`

## Docker
The base image is NVIDIA's `nvcr.io/nvidia/pytorch:24.02-py3` image. We overwrite the container's python install using the user defined dependencies. In some cases, creating a `venv` *inside* the dev/prod images may be preferred (multi-stage builds, isolation of host python, long CI/CD, etc.). See commented out lins in `Dockerfile.dev` for an illustrative example of how to setup .venv in the containers. It may be worth noting that since we are binding the host working directory to the dev container, the `venv` will be created in `~/<your-user-name>/build/` rather than the project working dir. 

## Digital Research Alliance of Canada Considerations
DRAC pre-builds many python packages into wheels that are stored in a local wheelhouse. It is best practice to use these wheels rather than use package distributions from PyPI. Therefore, consider pinning dependencies in `pyproject.toml` to match the pre-built wheels avaialble from DRAC. You can search the pre-built wheels with `avail_wheels` command on DRAC servers. 

Unfortunately, `poetry` [does not yet support use of a local directory as a wheel repository](https://github.com/python-poetry/poetry/issues/5983). Therefore, clone your `poetry` `venv` to a `requirements.txt` file by running this command:
```bash
poetry export --format "requirements.txt" --without-hashes --without-urls -vvv >> requirements.txt
```
For simplicity, a bash script for installing the project and dependencies is included, see: `./scripts/build_cc_venv.sh`. Simply run this script from the project working directory after cloning the project from github. You can also use the `./scripts/slurm/batch_build.sh` to submit the build as `slurm` batch job. 

## Tests
This repository uses `pytest`. Run tests using `pytest` -->
