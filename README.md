# comp-graph.lab3

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
