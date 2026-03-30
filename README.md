# Medical-Image-Project

## Quick Start
### train_alzheimer.py (train Alzheimer CNN)
```shell
python .\train_alzheimer.py --split-mode folders --split-root ..\split
```
### replay model result (replay 'train_alzheimer.py' terminal output)
```
python3 .\replay_model_result.py --run-dir outputs/alzheimer_run
```

## Project structure
```
project1/
├── sampleCode/
│   ├── outputs/
│   │   └── (run_name)/
│   │       ├── metrics.json
│   │       ├── split_summary.json
│   │       └── best_model.pt
│   ├── train_alzheimer.py
│   └── replay_model_result.py
└── split/
    ├── train/
    │   └── (class_name)/
    │       └── (image_name).jpg
    ├── val/
    │   └── (class_name)/
    │       └── (image_name).jpg
    ├── test/
    │   └── (class_name)/
    │       └── (image_name).jpg
    └── split_summary.json
```
