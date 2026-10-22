# Smarter Devices, Smaller Models: Unlocking Embedded AI Through Efficient Compression



> 💡 **Note for the team:** This is just a template. Update the above title with your AI Studio Challenge Project name. Remove all guidance notes and example text in this template and populate this README with your own content. You can work on this README throughout AI Studio, and get feedback from your AI Studio Coach and Challenge Advisor before finalizing it.  

---

### 👥 **Team Members**

| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|-------------------------------------------------------------------------------------------|
| Daniel Solares   | @danntdm      | Project Manager, Data exploration, visualization, overall project coordination            |
| Kyubin Hyun      | @hyunguy      | Data collection, exploratory data analysis (EDA), dataset documentation                   |
| Christina Peralta| @nina2k5      | Pipeline organization, cross‑team communication, and integration of analytical outputs    |
| Kavya Pokharel   | @kavyapokharel| Visualization, Model Training, Dataset, Overall                                                      |
| Alex Tamanaha| @alexdlct| Visualization, Model Training, Dataset, Overall                                                      |
| Sharanya Raj| @Sharanya-Raj| EDA, Visualization, GRU / CNN-GRU Model Implementation, Model Compression                                                      |



---

## 🎯 **Project Highlights**

**Latest Updates:**
All of us worked individually on initial exploratory data analysis and visualization, comparing results in order to ensure we were all on the right track.
We then proceeded to divide out different models to test in the following distribution:

- **Daniel**: CNN Model, Hybrid LTSM/GRU Model
    -   Validation Accuracy: 100%
- **Kavya**: CNN Model
    -   Validation Accuracy: 100%
- **Navya**: Small Conv1D (2 Layers) CNN
    -   Validation Accuracy: 93.64%
- **Sharanya**: GRU Model, CNN / GRU Hybrid Model with 1 Conv1D layer
    -   Validation Accuracy: 99.02%
- **Christina**: GRU Model
    -   Validation Accuracy: 66.67%
- **Alex**: LSTM Model
    -   Validation Accuracy: 98%

**Summary:**

- Developed baseline CNN, GRU, LSTM, and hybrid architectures for three-class bearing-fault classification from vibration signals.
- Explored class balance, signal quality, vibration amplitude, and impulsiveness using waveform plots, root-mean-square amplitude, and kurtosis.
- Implemented standardization using training-set statistics and applied the same transformation to validation and test data.
- Established baseline models for subsequent evaluation of classification accuracy, parameter storage, inference latency, and compression tradeoffs.

---

## 👩🏽‍💻 **Setup and Installation**

### Requirements

- MATLAB with Deep Learning Toolbox.
- Deep Learning Toolbox Model Compression Library for compression experiments and the estimateNetworkMetrics call in the CNN script.

The current implementation uses MATLAB. The repository's requirements.txt is currently empty.

### Clone the Repository

git clone https://github.com/Break-Through-Tech/MathWorks-1A-unlocking-embedded-ai-through-efficient-compression.git

Open the cloned folder in MATLAB and set it as the current folder.

### Explore and Preprocess the Data

Run the following commands in MATLAB from the repository root:

cd scripts
september_exploration
september_standardize

The standardization script computes the training mean and standard deviation and applies them to all three splits.

A prepared copy is included at data/standardized.mat. To regenerate that file, uncomment the save statement at the end of september_standardize.m.

### Train and Evaluate the CNN

While the current folder is scripts, run:

cnnTraining

The script trains a three-layer 1-D CNN, evaluates test accuracy, displays a confusion matrix, and estimates parameter storage.

### GRU Implementation

The GRU training script is located at models/uncompressed/GRUmodel.m.

It currently expects standardizedBearingData.mat with variables named trainDataStandardized, valDataStandardized, and testDataStandardized.

The supplied standardized dataset instead uses data/standardized.mat with variables named trainStandardized, valStandardized, and testStandardized.

These paths and variable names must be aligned before running the GRU script.

---

## 🏗️ **Project Overview**

**Describe:**

This project is part of the Fall 2026 Break Through Tech AI Studio program, in collaboration with MathWorks.

Our objective is to develop a vibration-based fault diagnosis system and investigate how neural network compression can make it suitable for embedded applications.

The classification task distinguishes normal bearings from inner-race and outer-race faults. Bearing-fault detection provides a practical setting for studying how sensor-based AI can support equipment monitoring.

Beyond classification accuracy, the project considers the constraints of small computing devices. We will evaluate how pruning, projection, and quantization affect model storage, computational requirements, and predictive performance.

---

## 📊 **Data Exploration**

**You might consider describing the following (as applicable):**

We use the Rolling Element Bearing Fault Diagnosis dataset attributed to Eric Bechhoefer. Dataset provenance and licensing are documented in data/README.md.

Each observation is a vibration signal containing 5,000 samples. The dataset documentation describes resampling to 48,828 Hz and segmentation into fixed-length windows.

| Split | Sequences | Samples per Sequence |
|-------|-----------|----------------------|
| Training | 393 | 5,000 |
| Validation | 27 | 5,000 |
| Test | 102 | 5,000 |

The training set contains 131 sequences from each class:

- Normal
- InnerRaceFault
- OuterRaceFault

**Potential visualizations to include:**

* Plots, charts, heatmaps, feature visualizations, sample dataset images
### Exploration and Preprocessing

Our exploratory workflow checks class counts, missing and infinite values, signal ranges, and representative waveforms.

We calculate root-mean-square amplitude and kurtosis to examine vibration magnitude and impulsiveness.

Inspection of the supplied training matrix found no missing or infinite values. Mean per-sequence RMS amplitude was approximately:

- Inner-race faults: 1.89
- Normal bearings: 0.78
- Outer-race faults: 0.88

These differences motivate examining both amplitude and temporal structure when developing classifiers.

Standardization uses a single mean and standard deviation computed from the training data only. The same values are applied to validation and test data, avoiding leakage of their statistics into preprocessing.
### Limitations

The project overview identifies the dataset as originating from a single test rig under controlled laboratory conditions.

Results therefore do not establish performance across different appliances, operating conditions, or sensor configurations. The small validation set also limits the precision of validation accuracy estimates.

---

## 🧠 **Model Development**

Team members explored convolutional, recurrent, and hybrid architectures in parallel.

### Shared CNN Implementation

The shared CNN training script uses:

- Three Conv1D layers with 8, 16, and 32 filters.
- Batch normalization and ReLU activations.
- Max pooling after the first two convolutional blocks.
- Global average pooling.
- A fully connected layer and softmax output for three-class classification.

Training uses Adam optimization, a mini-batch size of 32, and 30 epochs.

### Shared GRU Implementation

The shared GRU script uses:

- A single-channel sequence input.
- A 100-unit GRU using its final output.
- A fully connected classification layer and softmax output.

Training uses Adam optimization, a mini-batch size of 32, and 80 epochs.

### Additional Architectures

Additional team experiments include LSTM, CNN–GRU, and LSTM–GRU architectures.

Training configurations and evaluation outputs should be recorded alongside each model to support reproducible comparisons.

## Results and Key Findings

### Preliminary Team-Reported Results

| Contributor | Architecture Explored | Reported Accuracy |
|-------------|-----------------------|-------------------|
| Daniel | CNN and hybrid LSTM–GRU | 100% |
| Kavya | CNN | 100% |
| Navya | Small two-layer Conv1D CNN | 93.64% |
| Sharanya | GRU and one-layer Conv1D–GRU hybrid | 99.02% |
| Christina | GRU | 66.67% |
| Alex | LSTM | 98% |

These are preliminary team-reported results. Evaluation splits, model-to-score assignments, and checkpoint-selection procedures must be confirmed before treating these values as a controlled comparison of architectures.

---

## 📈 **Results & Key Findings**

Exploration indicates that vibration amplitude differs across fault classes. Waveform and kurtosis analysis provide additional ways to investigate temporal and impulsive behavior.

Compression outcomes and on-device performance have not yet been established.

The next readme update will include visuals to compare baseline and compressed models using:

- Classification accuracy, macro F1, and per-class recall.
- Confusion matrices.
- Learnable parameter count and parameter-storage estimates.
- Inference memory requirements.
- Inference latency under a consistent measurement setup.
- Relative reductions in storage and computation.

**You might consider describing the following (as applicable):**

* Performance metrics (e.g., Accuracy, F1 score, RMSE)
* How your model performed
* Insights from evaluating model fairness

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

---

## 🚀 **Next Steps**

1. Align dataset paths, variable names, and evaluation procedures across model scripts.
2. Confirm which reported scores correspond to validation or test data.
3. Measure baseline parameter storage and inference requirements.
4. Evaluate supported pruning, projection, and quantization techniques.
5. Use validation data to select compression settings and reserve test data for final evaluation.
6. Compare estimated Flash and RAM requirements against the STM32 NUCLEO-U575ZI-Q target described by our advisor.
7. Document accuracy–memory–latency tradeoffs and investigate hardware deployment as a stretch goal.

**You might consider addressing the following (as applicable):**

* What are some of the limitations of your model?
* What would you do differently with more time/resources?
* What additional datasets or techniques would you explore?

---

## 📝 **License**

Specify how your project can be used by others. Choose an appropriate license and link it here (e.g., MIT, Apache 2.0). Make sure your Challenge Advisor approves of the selected license type. 

**Example:**
This project is licensed under the MIT License.

---

## 📄 **References** (Optional but encouraged)

- Dataset documentation: data/README.md
- AI Studio project overview: Challenge-Project-Overview.md
- MathWorks bearing-fault diagnosis example:
  https://www.mathworks.com/help/predmaint/ug/rolling-element-bearing-fault-diagnosis-using-deep-learning.html
- MathWorks compression using projection:
  https://www.mathworks.com/help/deeplearning/ref/compressnetworkusingprojection.html
- MathWorks network metrics estimator:
  https://www.mathworks.com/help/deeplearning/ref/estimatenetworkmetrics.html
---

## 🙏 **Acknowledgements** (Optional but encouraged)

We thank MathWorks Challenge Advisor Nayara Aguiar, AI Studio Coach Rashidah Carr, and the Break Through Tech AI program for their guidance and support throughout the project.
