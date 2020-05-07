## Installation

```bash
conda create -n venv pip
conda activate venv

pip install -r requirements.txt
conda install -c pytorch torchvision cudatoolkit=9.0

export ROOT_DIR=$PWD

# Clone repository for VAE inference
git clone https://github.com/AntixK/PyTorch-VAE.git

# Download dlib’s pre-trained facial landmark detector
cd checkpoints
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

cd $ROOT_DIR
unset ROOT_DIR
```

## Usage

To run script, which modifies and saves image into `imgs` directory, use:
```python
python main.py -c configs/face_manipulation_cfg.yaml
```
To explore the logic internals and obtain detailed explanation of the method, use `demo.ipynb`
