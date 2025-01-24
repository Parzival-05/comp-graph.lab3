# comp-graph.lab3


[![Run python tests](https://github.com/Parzival-05/comp-graph.lab3/actions/workflows/tests.yaml/badge.svg)](
https://github.com/Parzival-05/comp-graph.lab3/actions/workflows/tests.yaml)
[![Code style & linting](https://github.com/Parzival-05/comp-graph.lab3/actions/workflows/code_style.yaml/badge.svg)](https://github.com/Parzival-05/comp-graph.lab3/actions/workflows/code_style.yaml)

## Get Started 

### Setup environment

```bash
pip install poetry
poetry install
```

### Download and unzip dataset

```bash
curl -L -o ./rice-image-dataset.zip https://www.kaggle.com/api/v1/datasets/download/muratkokludataset/rice-image-dataset
unzip rice-image-dataset.zip -d ./
```

Then you should have `Rice_Image_Dataset` directory with unpacked dataset.


### Run UI tool for inference

```
poetry run python3 app.py
```
Then go to http://127.0.0.1:5000/.

### Example:

<img src="./static/demo_example.png">
