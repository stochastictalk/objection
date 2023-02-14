# 🐊 Fortuna

Fortuna is a Python package for fitting object detection models

## Overview

This package allows you to fit an object detection model to your own data. 


## Priority stack

- Fix `Examples.ipynb`.
- Create small dummy dataset.
- Include data labelling instructions.
- End-to-end quickstart in readme.

## End-to-end Example

This example shows how to use the package to train an object detector using your own data. It goes through everything from creating image annotations through to evaluating a model and exporting it.

These instructions assume you have Docker installed.

### 1. Create image annotations using Label Studio.

We'll use[LabelStudio]() to create annotations for your images.

Move your images into a directory `<your-dataset-name>/images` and create an empty directory `<your-dataset-name>/masks`. Set `<your-dataset-name>` to your current working directory. Launch a local LabelStudio server using Docker via
```
docker run -it -p 8080:8080 -v ${PWD}:/label-studio/data heartexlabs/label-studio:latest
```
Open `localhost:8080` in your browser.

[Getting started with Label Studio](https://labelstud.io/blog/zero-to-one-getting-started-with-label-studio).

1. Create Project.
2. Import your data.
3. Select 'Semantic Segmentation with Polygons'.
4. Don't fuck about with the default labelling settings.
5. Use the lock if you need to label overlapping objects.
6. Click 'Submit' to save the annotation.
7. Navigate to the database view.

@TODO complete these instructions.


### 2. Load the data with `ObjectDetectionDataset`.



### 3. Fit the model with `ObjectDetectionModel.fit()`.



### 4. Evaluate the model with `ObjectDetectionModel.evaluate()`.



### 5. Export the model with `ObjectDetectionModel.write()`.



## Developer Quickstart

First, prove to yourself the package installs correctly, its tests run, and its documentation compiles.

If you're using Docker, launch a Python container and connect to its shell
```
docker run -it -p 3527:3527 -v ${PWD}:/package python:3.9 /bin/bash
```

Install the development version of the package in editable mode to the environment with
```
pip install -e '.[develop]'
```
The package's configuration is in `pyproject.toml` (summary [here](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html)). The directory structure follows the src layout ([details](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html)). Its version is managed using `setuptools_scm`, meaning version numbers are automatically extracted from git tags: you can read about the versioning logic [here](https://pypi.org/project/setuptools-scm/).

Install the precommit hooks:
```
pre-commit install
```
You can edit the config of these in `.pre-commit-config.yaml`.

Check the tests run:
```
pytest
```
Compile the documentation:
```
cd docs
make html
```
Host the resulting doc HTMLs using Python's webserver:
```
python -m http.server 3527 -d ../build/sphinx/html
``` 
Open a web browser on the host and go to `localhost:3527`. You should see the docs.

Finally, to create a wheel and sdist for your package:
```
python -m build --wheel
```
They will be output to a directory `dist/`.
