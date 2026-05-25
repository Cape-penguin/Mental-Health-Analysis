# Mental-Health-Analysis
This repo is for the paper "Mitigating Spurious Correlations in Mental Health Analysis: A Contrastive Learning Approach for Cross-Domain Robustness" submitted to [KDD Conference 2026](https://kdd2026.kdd.org/)

## Notice
- [*May. 10, 2026*] Our paper has been accepted to the KDD 2026 AI4S track.
- [*Feb. 8, 2026*] The repository has been fully organized and the source code is now officially uploaded.

## Project Structure
Short Document for Mental-Health Analysis.

```
root/
├── configs/
│   ├── config_bert.json        # hyperparameters for the BERT model.
│   ├── pretrain.yaml           # training hyperparameters.
│   └── sample-keywords.json    # pre-defined spurious correlation token IDs.
├── data/                       # storage for raw and processed text datasets in `.csv` format.
├── src/                    
│   ├── dataset/        # custom DataLoaders that parse CSV files and transform raw text.
│   ├── loss/           # implements the objective functions.
│   ├── model/          # contains the implementation of the proposed method.
│   ├── optim/          # manages the optimization logic (AdamW) used during the training process.
│   └── scheduler/      # contains learning rate schedulers.
└── Pretrain.py         # main execution script, including environment setup, model initialization, 
                          and the training loop based on the proposed methodology.
```

## Requirements
- torch 2.3.0
- transformer 4.41.1

## Pre-training
Run the following command to start the pre-training process. Once the training is complete, the resulting model checkpoints and outputs will be saved in the output directory (`./outputs/posts`).
```
python Pretrain.py --output_dir ./outputs/posts
```

## Reference
```
# The citation format will be updated once the paper is published in the ACM Digital Library.
```
