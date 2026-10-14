# HipMRI Prostate Segmentation with an Improved U-Net

## Overview

This project investigates 2D semantic segmentation of pelvic MRI slices from the HipMRI dataset using a standard U-Net baseline and an Improved U-Net.

The main objective is to determine whether the additional architectural complexity of the Improved U-Net provides a meaningful segmentation benefit over a standard U-Net, particularly for small and difficult anatomical structures such as the prostate.

The project follows the COMP3710 Normal Difficulty segmentation pathway:

- Dataset: HipMRI 2D MRI slices
- Proposed model: Improved U-Net
- Baseline: Standard 2D U-Net
- Target metric: prostate Dice score of at least 0.75 on the held-out test set
- Evaluation: Dice, IoU, qualitative error analysis, computational cost, and paired baseline comparison

The HipMRI prostate segmentation task is difficult because the prostate occupies a very small proportion of the pelvic MRI and has low-contrast boundaries near surrounding structures. These properties create severe class imbalance and make boundary accuracy important even when the global segmentation score appears high.

---

## Engineering Question

The main engineering question investigated in this project is:

> Does an Improved U-Net provide enough segmentation benefit over a standard 2D U-Net to justify its additional computational cost?

A second concern is whether a high global Dice score hides difficult local failures.

In prostate radiotherapy planning, segmentation errors near the prostate boundary may be more important than errors in large and easy regions. Therefore, this project evaluates not only aggregate Dice and IoU scores, but also individual difficult slices and paired model behaviour.

---

## Dataset

The project uses the HipMRI dataset supplied through the COMP3710 Rangpur HPC cluster.

Rangpur dataset location:

```text
/home/groups/comp3710/HipMRI_Study_open/keras_slices_data
```

The 2D dataset contains six folders:

```text
keras_slices_train
keras_slices_validate
keras_slices_test

keras_slices_seg_train
keras_slices_seg_validate
keras_slices_seg_test
```

MRI files use names such as:

```text
case_004_week_0_slice_0.nii.gz
```

The corresponding segmentation file uses:

```text
seg_004_week_0_slice_0.nii.gz
```

Therefore, MRI and segmentation files are paired by replacing the `case_` prefix with `seg_`.

All files use the NIfTI format (`.nii.gz`) and are loaded using `nibabel`.

---

## Dataset Inspection

A direct inspection of the Rangpur data showed:

```text
MRI shape:          (256, 128)
Segmentation shape: (256, 128)
Observed MRI range: 0.0 to 411.0
```

The segmentation masks contain six integer labels:

```text
[0, 1, 2, 3, 4, 5]
```

The training segmentation set contains:

```text
11,460 masks
```

A full scan of the training masks produced the following class statistics:

| Label | Total Pixels | Slices Containing Label |
|---|---:|---:|
| 0 | 64,897,313 | 11,460 |
| 1 | 253,682,431 | 11,460 |
| 2 | 46,951,538 | 11,406 |
| 3 | 7,296,009 | 8,300 |
| 4 | 1,340,454 | 4,056 |
| 5 | 1,599,295 | 4,152 |

These statistics demonstrate substantial class imbalance. Labels 4 and 5 are particularly small compared with the dominant regions.

The exact organ-to-label mapping was not explicitly provided in the files inspected on Rangpur. For this reason, the project does not assign anatomical names to numeric labels until the mapping is verified from an authoritative source.

TODO before final submission:

```text
Confirm which numeric label corresponds to the prostate.
```

This verified class ID will then be used when reporting the required prostate Dice score.

---

## Patient-Level Split

The supplied HipMRI split was explicitly checked for patient leakage.

The patient IDs were extracted from file names.

The resulting split was:

```text
Training patients:
004–035

Validation patients:
036–039

Test patients:
040–042
```

The following intersections were checked:

```text
Train / Validation overlap: set()
Train / Test overlap:       set()
Validation / Test overlap: set()
```

Therefore, there is no patient overlap between training, validation, and test sets.

This is important because different slices and treatment weeks from the same patient are strongly correlated. A slice-level random split could therefore produce artificially high test performance.

---

## Example Label Visualisation

A representative HipMRI slice was visualised together with the individual segmentation labels.

![HipMRI label visualisation](hipmri_label_map.png)

The visualisation confirms that the task is multi-class semantic segmentation and also illustrates the strong imbalance between large anatomical regions and small structures.

---

## Preprocessing

MRI slices are loaded using `nibabel` and converted to `float32`.

The original spatial size is retained:

```text
256 × 128
```

The images are not downsampled to `128 × 128` because the prostate and other small anatomical structures occupy relatively few pixels. Retaining the original 2D resolution avoids unnecessary loss of spatial information.

MRI intensity normalisation is performed using z-score normalisation:

```python
image = (image - image.mean()) / (image.std() + 1e-8)
```

Segmentation masks are converted to integer class IDs:

```python
mask = mask.astype(np.int64)
```

The model input and target shapes are:

```text
Image:
[1, 256, 128]

Mask:
[256, 128]
```

For a batch of size `B`:

```text
Input:
[B, 1, 256, 128]

Model output:
[B, 6, 256, 128]

Target:
[B, 256, 128]
```

The six output channels correspond to the six segmentation labels.

---

## Baseline Model

The baseline is a standard 2D U-Net.

The encoder progressively extracts higher-level features while reducing spatial resolution.

The decoder restores spatial resolution using transposed convolution and skip connections.

The main channel progression is:

```text
32
64
128
256
512
```

The U-Net uses skip connections between matching encoder and decoder stages.

The final layer is a `1 × 1` convolution that converts the final feature map into six class logits:

```text
output channels = 6
```

No softmax operation is applied inside the model because `CrossEntropyLoss` expects raw logits.

---

## Improved U-Net

The proposed model is an Improved U-Net.

It retains the U-Net encoder-decoder structure but introduces additional architectural components designed to improve feature learning.

The current implementation adds residual processing to the convolutional blocks and includes regularisation through dropout.

Residual connections allow the network to learn a residual mapping:

```text
output = F(x) + x
```

This can improve optimisation by allowing information and gradients to flow more directly through the network.

The main hypothesis is that the Improved U-Net may provide better representation of difficult anatomical regions, but at the cost of:

- more trainable parameters;
- higher training time;
- higher inference latency;
- potentially greater memory use.

The engineering comparison therefore evaluates whether the accuracy improvement is large enough to justify this extra complexity.

---

## Loss Function

The training objective combines pixel-wise classification with region-overlap optimisation.

### Cross-Entropy Loss

Cross entropy treats segmentation as per-pixel multi-class classification.

```python
cross_entropy = nn.CrossEntropyLoss()
```

It penalises incorrect class predictions at each pixel.

### Dice Loss

Dice loss directly measures overlap between predicted and ground-truth regions.

The Sørensen-Dice coefficient for a class is:

```text
Dice = 2 × |Prediction ∩ Ground Truth|
       --------------------------------
       |Prediction| + |Ground Truth|
```

A Dice score of:

```text
1.0
```

represents perfect overlap.

The combined loss allows the model to learn both class-level correctness and region-level overlap.

---

## Evaluation Metrics

The following metrics are reported on the held-out test set.

### Dice Score

Dice measures segmentation overlap and is the main metric used in this project.

Per-class Dice is reported for all six labels.

The final report also separately identifies the verified prostate label and reports:

```text
Prostate Dice
```

The Normal Difficulty target is:

```text
Prostate Dice ≥ 0.75
```

### Intersection over Union

IoU is calculated as:

```text
IoU = |Prediction ∩ Ground Truth|
      ---------------------------
      |Prediction ∪ Ground Truth|
```

Per-class IoU is reported together with Dice.

### Computational Cost

The following engineering costs are measured:

- trainable parameter count;
- training time;
- inference latency per image;
- peak GPU VRAM when running on Rangpur.

These measurements are used to determine whether the Improved U-Net provides a practical improvement over the baseline.

---

## Experimental Setup

Both models are evaluated using the same:

- training split;
- validation split;
- test split;
- image resolution;
- batch size;
- optimiser;
- learning rate;
- loss function;
- evaluation metrics.

This ensures that the architecture is the main variable being compared.

Current intended settings:

```text
Input resolution: 256 × 128
Number of classes: 6
Optimizer: Adam
Learning rate: 0.001
Loss: Cross Entropy + Dice
```

TODO after final HipMRI training:

```text
Final batch size:
Final epoch count:
Rangpur GPU:
Peak GPU VRAM:
```

---

## Training Procedure

For every batch:

```text
MRI images
    ↓
forward pass
    ↓
six-channel segmentation logits
    ↓
Cross-Entropy + Dice loss
    ↓
backpropagation
    ↓
Adam update
```

The model is evaluated on the validation set after every epoch.

The checkpoint with the best validation Dice is saved.

This is preferable to automatically using the last epoch because continued reduction in training loss does not guarantee improved validation performance.

---

## Baseline Results

TODO: replace this section with the final HipMRI baseline experiment.

Expected table format:

| Metric | Basic U-Net |
|---|---:|
| Test Loss | TODO |
| Mean Dice | TODO |
| Mean IoU | TODO |
| Prostate Dice | TODO |
| Prostate IoU | TODO |
| Parameters | TODO |
| Inference Time | TODO |
| Peak GPU VRAM | TODO |

Per-class results:

| Label | Dice | IoU |
|---|---:|---:|
| 0 | TODO | TODO |
| 1 | TODO | TODO |
| 2 | TODO | TODO |
| 3 | TODO | TODO |
| 4 | TODO | TODO |
| 5 | TODO | TODO |

---

## Improved U-Net Results

TODO: replace this section with the final HipMRI Improved U-Net experiment.

| Metric | Improved U-Net |
|---|---:|
| Test Loss | TODO |
| Mean Dice | TODO |
| Mean IoU | TODO |
| Prostate Dice | TODO |
| Prostate IoU | TODO |
| Parameters | TODO |
| Inference Time | TODO |
| Peak GPU VRAM | TODO |

Per-class results:

| Label | Dice | IoU |
|---|---:|---:|
| 0 | TODO | TODO |
| 1 | TODO | TODO |
| 2 | TODO | TODO |
| 3 | TODO | TODO |
| 4 | TODO | TODO |
| 5 | TODO | TODO |

---

## Baseline versus Improved U-Net

The final comparison will use the same held-out HipMRI test split.

TODO after final experiments:

| Model | Mean Dice | Prostate Dice | Mean IoU | Parameters | Inference Time | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Basic U-Net | TODO | TODO | TODO | TODO | TODO | TODO |
| Improved U-Net | TODO | TODO | TODO | TODO | TODO | TODO |

The important question is not simply whether the Improved U-Net achieves a numerically higher Dice score.

The comparison must determine:

> Is the improvement large and consistent enough to justify the additional computational cost?

---

## Paired Slice-Level Comparison

A paired comparison is performed by running both models on exactly the same test MRI slices.

For each slice:

```text
delta Dice =
Improved U-Net foreground Dice
-
Basic U-Net foreground Dice
```

This allows identification of:

```text
cases where the Improved U-Net helps most
```

and:

```text
cases where the Improved U-Net performs worse
```

This is useful because a small improvement in global average Dice may hide large improvements or degradations on individual difficult slices.

TODO after HipMRI training:

```text
Number of test slices:
Improved better:
Improved worse:
Mean paired Dice difference:
Largest positive difference:
Largest negative difference:
```

---

## Failure Case Analysis

Global Dice alone is insufficient for evaluating prostate segmentation.

The most important failures are likely to occur in:

- small prostate regions;
- low-contrast boundaries;
- prostate base slices;
- prostate apex slices;
- regions adjacent to the bladder or rectum;
- slices where only a very small foreground structure is present.

The final analysis will inspect at least 3–5 representative failure cases.

Each failure visualisation will contain:

```text
MRI
Ground Truth
Prediction
Error Map
```

The error map highlights pixels where the predicted class differs from the ground-truth segmentation.

TODO after final inference:

### Failure Case 1

```text
Slice:
Failure type:
Prostate Dice:
Observed error:
Likely explanation:
Operational implication:
```

### Failure Case 2

```text
Slice:
Failure type:
Prostate Dice:
Observed error:
Likely explanation:
Operational implication:
```

### Failure Case 3

```text
Slice:
Failure type:
Prostate Dice:
Observed error:
Likely explanation:
Operational implication:
```

The purpose of the analysis is to determine whether a high aggregate metric hides clinically important local errors.

---

## Qualitative Results

Example prediction format:

```text
MRI | Ground Truth | Prediction | Error Map
```

TODO:

```markdown
![HipMRI prediction example](hipmri_prediction.png)
```

Paired comparison examples:

```markdown
![Improved model better case](paired_better_1.png)

![Improved model worse case](paired_worse_1.png)
```

These figures will be added after the final HipMRI models are trained.

---

## Computational Cost

The project evaluates the accuracy-cost trade-off.

TODO after Rangpur profiling:

| Model | Parameters | Training Time | Inference / Image | Peak GPU VRAM |
|---|---:|---:|---:|---:|
| Basic U-Net | TODO | TODO | TODO | TODO |
| Improved U-Net | TODO | TODO | TODO | TODO |

GPU memory will be measured on Rangpur rather than estimated.

No synthetic or guessed GPU memory values are reported.

---

## Engineering Recommendation

TODO: update this section using the final HipMRI results.

The final recommendation will consider both segmentation quality and computational cost.

A possible decision rule is:

```text
Use the Improved U-Net only if it provides a meaningful and consistent
improvement in prostate boundary segmentation relative to the standard U-Net.
```

If the improvement is only marginal while computation increases substantially, the standard U-Net may be preferable for a fast annotation-assistance workflow.

If the Improved U-Net produces a meaningful improvement on difficult prostate slices without excessive additional cost, it may be preferred as the research model.

However, even if the target Dice threshold is achieved, difficult or low-confidence cases should still be routed to a human specialist rather than accepted automatically.

This is especially important for base and apex slices, where a global segmentation score may conceal significant local contour errors.

---

## Project Structure

```text
Improved_UNet_XuanyuLi/
│
├── modules.py
├── dataset.py
├── train.py
├── predict.py
├── compare_models.py
├── README.md
│
└── result figures
```

### `modules.py`

Contains:

```text
BasicUNet
ImprovedUNet
supporting convolution and residual modules
```

### `dataset.py`

Responsible for:

```text
loading NIfTI MRI files
loading segmentation masks
pairing case and segmentation files
normalising MRI intensity
creating train/validation/test DataLoaders
preserving the official patient-independent split
```

### `train.py`

Responsible for:

```text
training
validation
loss computation
Dice
IoU
checkpointing
runtime profiling
```

### `predict.py`

Responsible for:

```text
loading a saved checkpoint
test inference
qualitative segmentation visualisation
error-map generation
failure-case extraction
```

### `compare_models.py`

Responsible for:

```text
paired Basic U-Net versus Improved U-Net comparison
per-slice Dice comparison
identifying improvement cases
identifying degradation cases
```

---

## Usage

### Dataset

The HipMRI dataset is not included in this repository.

On Rangpur, the expected dataset root is:

```text
/home/groups/comp3710/HipMRI_Study_open/keras_slices_data
```

### Training the Baseline

Example:

```bash
python train.py \
    --data-root /home/groups/comp3710/HipMRI_Study_open/keras_slices_data \
    --model baseline \
    --epochs TODO \
    --batch-size TODO
```

### Training the Improved U-Net

```bash
python train.py \
    --data-root /home/groups/comp3710/HipMRI_Study_open/keras_slices_data \
    --model improved \
    --epochs TODO \
    --batch-size TODO
```

### Inference

```bash
python predict.py \
    --data-root /home/groups/comp3710/HipMRI_Study_open/keras_slices_data \
    --checkpoint best_improved_unet.pth \
    --model improved
```

### Paired Model Comparison

```bash
python compare_models.py \
    --data-root /home/groups/comp3710/HipMRI_Study_open/keras_slices_data \
    --baseline-checkpoint best_baseline_unet.pth \
    --improved-checkpoint best_improved_unet.pth
```

---

## Dependencies

The project requires:

```text
Python
PyTorch
torchvision
NumPy
nibabel
matplotlib
```

TODO before submission:

Run:

```bash
python -c "
import sys
import torch
import torchvision
import numpy
import nibabel
import matplotlib

print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('NumPy:', numpy.__version__)
print('nibabel:', nibabel.__version__)
print('matplotlib:', matplotlib.__version__)
"
```

Then replace this section with the exact versions used for the final experiment.

---

## Reproducibility

The following steps are used to improve reproducibility:

- fixed patient-independent train/validation/test split;
- identical preprocessing for the baseline and proposed model;
- identical evaluation split;
- explicit model checkpoints;
- reported software versions;
- documented training settings;
- deterministic random seed where applicable.

The dataset and trained `.pth` checkpoints are intentionally not committed to the Git repository.

---

## Data Leakage Verification

The following patient-level check was performed directly on the Rangpur dataset:

```text
Train/Val overlap: set()
Train/Test overlap: set()
Val/Test overlap: set()
```

This confirms that no patient occurs in more than one dataset partition.

This is especially important because HipMRI contains multiple slices and treatment weeks for individual patients.

---

## Limitations

The current project uses a 2D slice-wise approach.

This provides lower memory usage and simpler training compared with a full 3D model, but it does not explicitly model continuity between adjacent slices.

As a result, possible limitations include:

- inconsistent contours between neighbouring slices;
- reduced context at the prostate base and apex;
- sensitivity to small foreground structures;
- class imbalance;
- errors near low-contrast anatomical boundaries.

A future extension could investigate a 3D U-Net or other spatial-context model.

---

## Artificial Intelligence Usage Disclosure

OpenAI ChatGPT was used as an engineering and learning assistant during this project.

AI assistance was used for:

- discussing U-Net and residual architecture concepts;
- explaining PyTorch segmentation operations;
- scaffolding portions of the implementation;
- debugging data-loading and command-line issues;
- designing evaluation procedures;
- developing paired baseline comparison logic;
- suggesting failure-case analysis procedures;
- reviewing the structure and wording of project documentation.

All AI-assisted content was manually inspected and empirically tested.

Verification included:

- checking MRI and segmentation tensor shapes;
- inspecting segmentation label values;
- verifying MRI-mask file pairing;
- checking the complete training label set;
- verifying patient-level train/validation/test separation;
- running baseline and Improved U-Net models;
- comparing models on identical evaluation data;
- visualising predictions and segmentation errors;
- checking reported metrics against program output.

The author remains responsible for all implementation decisions, experiments, reported values, interpretations, and conclusions.

---

## References

1. COMP3710 Pattern Recognition and Analysis, 2026 Project Specification.
2. Ronneberger, O., Fischer, P., and Brox, T.  
   U-Net: Convolutional Networks for Biomedical Image Segmentation.
3. HipMRI Study dataset supplied through the COMP3710 Rangpur HPC environment.
4. Additional references for the Improved U-Net architecture will be added according to the final implementation.

---

## Current Project Status

Completed:

```text
✓ GitHub project and topic-recognition branch
✓ Basic U-Net implementation
✓ Improved U-Net implementation
✓ Training pipeline
✓ Validation pipeline
✓ Dice and IoU evaluation code
✓ Inference visualisation
✓ Failure-case extraction
✓ Paired model comparison
✓ Rangpur access
✓ HipMRI data discovery
✓ NIfTI loading
✓ MRI-mask pairing
✓ Six-class mask verification
✓ Patient-level leakage verification
✓ Label-distribution analysis
```

Remaining:

```text
□ Verify the numeric prostate label
□ Update dataset.py for HipMRI NIfTI input
□ Confirm six-class model output
□ Run HipMRI Basic U-Net baseline
□ Run HipMRI Improved U-Net
□ Achieve prostate Dice >= 0.75
□ Profile Rangpur GPU VRAM
□ Generate final HipMRI figures
□ Perform HipMRI failure-case analysis
□ Fill all TODO result tables
□ Add exact dependency versions
□ Push final commits
□ Open upstream pull request
□ Export README to PDF
□ Submit PDF to Turnitin
```