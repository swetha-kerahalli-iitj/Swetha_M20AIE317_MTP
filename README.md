## Introduction

**LEARN Codes: Inventing Low-latency Codes via Recurrent Neural Networks**

## How To Run

```bash
git clone https://github.com/CruiseTian/LEARN.git && cd LEARN

conda env create -f environment.yaml
# or 
conda create -n pytorch python=3.8.3   
source activate pytorch      
pip install -r requirements.txt 

# program start
python main.py -num_epoch 120
```

`main.py get_args.py`。

## File structure

logs_test -- log files

tmp -- temporary path

data_test -- data files

model_test -- model files

## Plan
- [x] attention