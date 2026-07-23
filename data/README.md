# Rolling Element Bearing Fault Diagnosis Dataset

## Source

Data owner: [Eric Bechhoefer](mailto:eric@gpms-vt.com)  
Original source: data-acoustics.com/measurements/bearing-faults/bearing-2

## License

[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Files

| File | Contents |
|------|----------|
| `train.mat` | 393 samples (131 per class) |
| `val.mat` | 27 samples |
| `test.mat` | 102 samples |

Each file contains two variables:
- **`trainData` / `valData` / `testData`** — (N x 5000) float64 matrix. Each row is one vibration signal segment.
- **`trainLabels` / `valLabels` / `testLabels`** — (N x 1) cell array of strings: `"Normal"`, `"InnerRaceFault"`, or `"OuterRaceFault"`.

## How to Load

**Python:**
```python
import scipy.io

train = scipy.io.loadmat('train.mat')
X_train = train['trainData']       # shape: (393, 5000)
y_train = [label[0] for label in train['trainLabels'].flatten()]
```

**MATLAB:**
```matlab
load('train.mat')  % gives trainData and trainLabels
```

## Modifications from Original Dataset

The original dataset contains 6 folders: baseline conditions, outer race faults, inner race faults, analyses, and real-world examples. For this project:

- Only the first 4 folders are used (baseline, outer race, inner race fault conditions)
- Signals resampled to a consistent 48,828 Hz sample rate
- Segmented into fixed-length windows of 5,000 samples
- Balanced across the 3 classes
- Split into train (75%), validation (5%), and test (20%) sets

## Remaining Preprocessing

The data is ready to use except for **standardization**, which should be applied before model training.
