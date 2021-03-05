# Image classification with PyTorch

Training classification NN on [Imagewoof](https://github.com/fastai/imagenette).

### 1. Fork / Clone this repository

```bash
git clone https://github.com/7CD/image_classification_pytorch.git
cd image_classification_pytorch
```
 
### 2. Create and activate virtual environment

Create virtual environment `myvenv`
```bash
python3 -m venv myvenv
echo "export PYTHONPATH=$PWD" >> myvenv/bin/activate
source myvenv/bin/activate
```
Install python libraries

```bash
pip install -r requirements.txt
```
Add Virtual Environment to Jupyter Notebook

```bash
python3 -m ipykernel install --user --name=myvenv
``` 

### 3. Download Imagewoof dataset

```bash
python3 src/data_load.py --config params.yaml --size full
```

### 4. Train model

following [notebook](https://github.com/7CD/image_classification_pytorch/blob/main/notebooks/baseline.ipynb) or by command:

```bash
python3 resnet50_train.py --model-name baseline --lr 0.001 --epochs 1 --gpu
```
(Accuracy on Imagewoof = 0.96)
