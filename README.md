# Medical-Image-Project

## Quick Start
### train_alzheimer.py (train Alzheimer CNN)
```
python3 .\train_alzheimer.py --split-mode folders --split-root ..\split --output-dir outputs/<run_name>
# or
python3 .\train_alzheimer.py --split-mode folders --train-dir ..\split\train_double --val-dir ..\split\val --test-dir ..\split\test --output-dir outputs/<run_name>
```
### replay model result (replay 'train_alzheimer.py' terminal output)
```
python3 .\replay_model_result.py --run-dir outputs/<run_name>
```
### [tool] augmentation train (augment train set by "horizontal flip")
```
python3 .\augmentation_train.py
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

## Test Accuracy record
| Method    | Test1     | Test2     | Test3     | Average   |
| :---:     | :---:     | :---:     | :---:     | :---:     |
| baseline  | 0.590625  | 0.55625   | 0.5875    | 0.578125  |
| AvgPool   | 0.5421875 | 0.5640625 | 0.5203125 | 0.5421875 |
| LeakyReLU | 0.5671875 | 0.571875  | 0.5828125 | 0.5739583 |
| augment   | 0.578125  | 0.5703125 | 0.5734375 | 0.5739583 |
