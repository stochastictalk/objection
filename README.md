# ❗ Objection

`objection` is a Python package that simplifies the workflow for creating a semantic segmentation/object detection model. 

It includes
- ✏️ An out-of-the-box solution for creating image masks.
- 🏋️ Painless loading of object detection data into PyTorch.
- ⚡ Automated fine-tuning of a model sized for your use-case.
- 🧪 Convenience functions for in-depth validation of the trained model on out-of-sample data.

@TODO:
- Integrate LabelStudio.
- Write instructions for end-to-end example.
- Tools for model validation.

## End-to-end Example

This example shows how to use the package to train an  using your own data. It goes through everything from creating image annotations through to evaluating a model and exporting it. It assumes you are using MacOS or Linux.

@TODO: toc

### 1. Configure Label Studio.

We'll use[LabelStudio]() to create annotations for your images. Launch it using Docker by running
```
docker run -it -p 8080:8080 -v ${PWD}/mydata:/label-studio/data heartexlabs/label-studio:latest label-studio --log-level DEBUG
``` 
If you are using Apple silicon then you will need to configure Docker to use architecture emulation by setting the environment variable `DOCKER_DEFAULT_PLATFORM=linux/amd64`.

Open `localhost:8080` in your browser.

Create an account. The account details are only stored locally.

Click 'Create Project', give your project a name, then click 'Data Import'. 

Upload the files from your machine that you want to create labels for. You can Shift+click to select a range. When you're done, click 'Labeling Setup' and choose 'Semantic Segmentation with Polygons'.

Remove the default labels and add the class labels you plan to use. In this example, there is only one class - 'Hat'. Click 'Save' to complete setup.

### 2. Create and Export Image Annotations.

To label an image, click its entry in the navigation view, select the label beneath the image of the object you want to label, then click out a sequence of points that enclose the object. 

To label multiple objects, especially if they overlap, you'll find it useful to use the 'lock' and 'hide' features available when you highlight an entry in the central panel. Click 'Submit' to save your annotations.

@TODO: re-loading data from a previous session (probably just restarting the docker container).

When you've created all of your labels, export them via the 'Export' button. Select the 'COCO' format then click 'Export'. Your browser will download a zip archive.

[Getting Started with Label Studio](https://labelstud.io/blog/zero-to-one-getting-started-with-label-studio).


### 2. Load the data with `ObjectDetectionDataset`.

Unzip the archive and copy the filepath of `results.json`. Be sure not to separate `results.json` and the `images/` directory - you can move them, but make sure they sit side-by-side.

Launch a Python interpreter and load the data as a PyTorch dataset:
```
from objection import ObjectDetectionDataset

data = ObjectDetectionDataset("data/hats/result.json")
```


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
