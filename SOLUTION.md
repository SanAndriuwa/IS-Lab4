# Lab4: MATLAB solutions

The original README.md is unchanged. The supplied upstream MATLAB examples are preserved in `original/` for reference. The new solution uses small custom MATLAB functions, which the assignment permits, and requires only base MATLAB (R2016b or later).

## Folders and run order

- `data/INSTRUCTIONS.md`: prepare your own handwriting images here first.
- `main/lab4_rbf.m`: compare exactly 13 and 8 RBF neurons.
- `additional/lab4_mlp.m`: classify the same digits with a 35-20-10 MLP.
- `shared/`: feature extraction, data loading and network functions used by the scripts.

Download the whole repository so the shared functions and data folder remain together. Open a main/additional script in MATLAB and press Run. Paths are resolved from the script location. Both scripts clear the workspace and close figures.

## Images and features

Provide two training rows and one separately handwritten test row, each in the order `0123456789`. This gives 20 training examples and 10 test examples. A 13-center RBF needs at least 13 training examples, so a single training row is insufficient. Read the image instructions before photographing.

Processing: grayscale → threshold → separate rows/digits by blank gaps → crop → resize to 70x50 → split into 7x5 blocks → mean ink density per block. Each digit becomes a 35-element column. Always inspect the segmentation figures before interpreting results. The simple method is intended for clean, well-lit images with separated strokes and digits; it is not a general OCR system.

## RBF main task

Use width 1 for both networks. Centers are training examples selected by farthest-point selection: start with the first example and repeatedly choose the example farthest from existing centers. Output weights and biases are fitted by least squares with small regularization (0.0001). This is a custom RBF implementation, not the exact greedy algorithm used by MATLAB newrb. The number of centers is explicitly 13 or 8. Ten outputs represent digits 0-9; choose the largest output. RBF scores are not normalized probabilities.

Compare the printed training/test accuracies and per-digit predictions for 13 versus 8 neurons. Fewer neurons need not always be worse; describe the actual results. Do not select hyperparameters using the test image and then claim it remained an independent test.

## MLP additional task

The MLP has 35 inputs, 20 tanh hidden neurons and 10 softmax outputs. It uses full-batch backpropagation, cross-entropy loss, 3000 epochs and learning rate 0.1. Random seed 4 makes MATLAB runs repeatable. Softmax is stabilized by subtracting the largest logit. The hidden gradient is calculated using old output weights. Both networks receive the exact same extracted features and training/test split.

## Validation and remaining work

All new MATLAB files passed the MATLAB R2026a standalone static analyzer. Full MATLAB startup failed before execution with `File system inconsistency`. Independent Python checks confirmed finite outputs and learning on artificial feature vectors, and checked an MLP gradient by finite differences. Artificial vectors are only a math check, not handwritten recognition results.

No handwriting images or recognition accuracy claims are supplied. Add your own images, inspect segmentation, run both scripts, and record actual accuracies before submitting the lab. Personal defense notes are in Notion.
