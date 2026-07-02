# HTAN Combo Loss Experiment Results

## Experiment

This experiment tests whether the Taghanaki et al.-inspired Combo Loss improves HTAN on
ISIC-2018 skin lesion segmentation.

The comparison uses the same model and validation split:

- Model: `htan_2_n2`
- Dataset: ISIC-2018
- Split: seed `123`, 2074 train images and 520 validation images
- Image size: `256x256`
- GPU: NVIDIA A10G

## Losses Compared

Original HTAN loss:

```text
0.5 * BCEWithLogits + 0.5 * Dice
```

Combo Loss:

```text
alpha = 0.7
beta = 0.7
overlap = dice
```

The Combo Loss uses weighted cross entropy as the main signal and Dice as an overlap
regularizer. Since `beta > 0.5`, false negatives are penalized more strongly.

## ISIC-2018 Results

| Loss | Dice | IoU | ACC | REC | PRE |
|---|---:|---:|---:|---:|---:|
| Original BCE + Dice | **90.66** | **83.46** | **96.18** | 89.61 | **92.78** |
| Combo Loss alpha=0.7 beta=0.7 | 90.37 | 82.95 | 96.00 | **89.93** | 91.77 |

## Interpretation

The original HTAN loss performed better overall on Dice, IoU, accuracy, and precision.
The Combo Loss improved recall from `89.61` to `89.93`, which matches the hypothesis that
false-negative weighting can increase sensitivity to lesion pixels.

However, the recall improvement came with lower precision, suggesting more false positives.
Under this setting, Combo Loss is useful as a recall-oriented experiment, but it does not
replace the original HTAN loss as the best overall ISIC configuration.

## Stability Note

The first Combo Loss run reached a best validation Dice of `90.37` around epoch 35, but
later became unstable and produced `nan` losses. The best checkpoint was preserved and
evaluated. A more conservative Combo Loss setting may be worth testing next.

Recommended next experiment:

```bash
python3 train.py --model htan_2_n2 --dataset isic --loss combo --combo-alpha 0.8 --combo-beta 0.6 --combo-overlap dice
```

This reduces the false-negative weighting and increases the smooth cross-entropy component,
which may keep the recall benefit while reducing false positives and instability.
