### Setup Instructions

1. **Extract the dataset**  
   Unzip the dataset archive into the `data/` directory:

   ```bash
   unzip data.zip -d data
   ```

2. **Install dependencies**  
   Install all required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

> Make sure you're using **Python 3.9** or higher.

### Best configurations

#### YAGO6K103

transe:
`python train.py -d yago_new -m transe -e 1000 -ed 1024 -nen 128 -b 1024 -lo nssa_loss -lm 9 -lr 0.0001`

rotate:
`python train.py -d yago_new -m rotate -e 1000 -ed 256 -nen 256 -b 1024 -lo nssa_loss -lm 9 -lr 0.001`

cst:
`python train.py -d yago_new -m cs-transe -e 1000 -ed 512 -ced 128 -nen 32 -nenT 32 -b 1024 -lo nssa_loss -lm 9 -i_per 0.8 -lr 0.02 -lr_kappa 0.02 -lr_beta 0.3 -lr_eta 0.002`

csr:
`python train.py -d yago_new -m cs-rotate -e 1000 -ed 256 -ced 64 -nen 512 -nenT 32 -b 1024 -lo nssa_loss -lm 9 -i_per 0.8 -lr 0.02 -lr_kappa 0.02 -lr_beta 0.2 -lr_eta 0.01`

#### NELL995

transe:
`python train.py -d NELL-995_new -m transe -e 1000 -ed 256 -nen 512 -b 1024 -lo nssa_loss -lm 9 -lr 0.0001`

rotate:
`python train.py -d NELL-995_new -m rotate -e 1000 -ed 512 -nen 512 -b 1024 -lo nssa_loss -lm 9 -lr 0.001`

cst:
`python train.py -d NELL-995_new -m cs-transe -e 1000 -ed 512 -ced 64 -nen 512 -nenT 32 -b 1024 -lo nssa_loss -lm 9 -i_per 0.8 -lr 0.001 -lr_kappa 0.001 -lr_beta 0.01 -lr_eta 0.01`

csr:
`python train.py -d NELL-995_new -m cs-rotate -e 1000 -ed 512 -ced 128 -nen 512 -nenT 32 -b 1024 -lo nssa_loss -lm 9 -i_per 0.8 -lr 0.005 -lr_kappa 0.005 -lr_beta 0.05 -lr_eta 0.01`

#### FB15K237

transe:
`python train.py -d FB_new -m transe -e 1000 -ed 1024 -nen 512 -b 1024 -lo nssa_loss -lm 9 -lr 0.001`

rotate:
`python train.py -d FB_new -m rotate -e 1000 -ed 512 -nen 512 -b 1024 -lo nssa_loss -lm 9 -lr 0.001 -at 2.0`

cst:
`python train.py -d FB_new -m cs-transe -e 1000 -ed 512 -ced 512 -nen 512 -nenT 64 -b 1024 -lo nssa_loss -lm 9 -i_per 0.8 -lr 0.005 -lr_kappa 0.01 -lr_beta 0.5 -lr_eta 0.005 -at 2.0`

csr:
`python train.py -d FB_new -m cs-rotate -e 1000 -ed 1024 -ced 256 -nen 512 -nenT 512 -b 1024 -lo nssa_loss -lm 9 -i_per 0.7 -lr 0.005 -lr_kappa 0.001 -lr_beta 0.05 -lr_eta 0.01 -at 1.5`

#### DB241

transe:
`python train.py -d DB_new -m transe -e 1000 -ed 256 -nen 512 -b 1024 -lo nssa_loss -lm 18 -lr 0.001`

rotate:
`python train.py -d DB_new -m rotate -e 1000 -ed 512 -nen 512 -b 1024 -lo nssa_loss -lm 15 -lr 0.002`

cst:
`python train.py -d DB_new -m cs-transe -e 1000 -ed 512 -ced 64 -nen 512 -nenT 32 -b 1024 -lo nssa_loss -lm 15 -i_per 0.8 -lr 0.001 -lr_kappa 0.001 -lr_beta 0.1 -lr_eta 0.01 -at 2.0`

csr:
`python train.py -d DB_new -m cs-rotate -e 1000 -ed 1024 -ced 128 -nen 512 -nenT 16 -b 1024 -lo nssa_loss -lm 18 -i_per 0.8 -lr 0.005 -lr_kappa 0.001 -lr_beta 0.05 -lr_eta 0.01`
