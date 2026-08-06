# Smarter Devices, Smaller Models: Unlocking Embedded AI Through Efficient Compression

**Company / Org:** MathWorks  
**Challenge Advisors:** Nayara Aguiar, [Email address]  
**AI Studio Coach:** Rashidah Carr, rashidah.carr@breakthroughtech.org     
**Program:** Break Through Tech AI Studio - Fall 2026

---

## 🏢 About MathWorks

MathWorks is a leader in technical computing and model-based design. We provide software solutions that empower engineers and scientists to create and analyze complex systems. Our tools are key in industries such as automotive, aerospace, and communications.

---

## 🎯 The Challenge

### Project Summary
In this project, you will use sensor data and deep learning to develop a fault diagnosis system for smart appliances, optimized for embedded applications. You will learn how to build, train, and compress AI models while gaining hands-on experience with industry tools and providing insights about their usability.

### Success Criteria
- Development of a deep learning model for a smart appliance fault diagnosis application
- Performance analysis including inference speed, memory footprint, and accuracy
- Application of at least two model compression techniques with documented trade-offs

### Project Milestones

Use these milestones to guide your work. Your team will create a **GitHub Projects board** to track tasks within each milestone.

| Month | Milestone | Key Activities |
| :--- | :--- | :--- |
| September | Data Pipeline, Preprocessing & Baseline Model | • Ingest and preprocess rolling element bearing time-series vibration data (.mat format).<br>• Perform signal processing and feature extraction (time-domain, frequency-domain / FFT).<br>• Train baseline deep learning classifier (e.g., 1D-CNN or ResNet) for bearing fault diagnosis.<br>• Establish baseline metrics (Accuracy, F1-Score, Model Footprint in MB, FLOPs). |
| October | Model Compression & Optimization | • Apply structured and unstructured pruning to eliminate redundant neural network weights.<br>• Implement Post-Training Quantization (PTQ) or Quantization-Aware Training (QAT) to convert float32 models to int8.<br>• Experiment with Knowledge Distillation to transfer fault detection capability to a compact student network.<br>• Profile accuracy vs. memory footprint trade-offs across compression techniques. |
| November / December | Profiling, Interactive UI & Capstone Deliverables | • Evaluate inference latency, RAM footprint, and diagnostic accuracy under simulated edge resource constraints.<br>• Develop an interactive Streamlit application to visualize raw signal inputs, fault predictions, and model compression statistics.<br>• Finalize clean, reproducible GitHub repository, documentation, and stakeholder presentation deck. |

### Stretch Goals
* **Microcontroller / Edge Hardware Deployment:** Deploy compressed models onto physical edge microcontrollers or single-board computers (e.g., Raspberry Pi or STM32) and profile real-world execution latency.
* **Real-Time Streaming Anomaly Detector:** Build a continuous stream-processing pipeline that ingests live vibration sensor data and detects early bearing degradation in real time.
* **Cross-Operating Condition Generalization:** Benchmark compressed model robustness across varying rotational speeds, load conditions, and unseen bearing fault severities.

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset

**Name:** Rolling Element Bearing Fault Diagnosis  
**Format:** MATLAB (.mat), can be read in Python using `scipy.io.loadmat()` or loaded directly in MATLAB  
**Size:** Under 1 GB  
**Location:** [data folder](data) (see `data/README.md` for full details on source, license, and how to load)

### Key Details
- Vibration acceleration data from rolling element bearings
- Three fault classes: **Normal** (healthy), **Outer Race Fault**, and **Inner Race Fault**
- Pre-split into train, validation, and test sets (balanced, segmented into 5,000-sample windows)
- The remaining preprocessing step is **standardization** before model training
- Known limitation: Data comes from a single test rig under controlled laboratory conditions.

---

## 🛠️ Suggested Approach

**ML Problem Type:** Classification (fault type identification from vibration signals)

**Recommended Tools & Libraries:**

| Category | Python Options | MATLAB Options |
|----------|---------------|----------------|
| Data exploration | scipy, numpy, pandas, matplotlib | MATLAB built-in functions and graphics |
| Deep Learning | PyTorch, TensorFlow/Keras | Deep Learning Toolbox |
| Model Compression | torch.quantization, torch.nn.utils.prune, TFLite | Deep Learning Toolbox Model Compression Library |
| Environment | Google Colab | MATLAB Online or MATLAB Desktop |

**Evaluation Metrics:**
- Classification Accuracy
- Model Size (MB)
- Compression Ratio (original vs. compressed)
- Inference Speed (ms per sample)

---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Videos:**
- [A Practical Introduction to Edge AI](https://www.youtube.com/watch?v=ibm6ZRi6Sm4)
- [Compressing Neural Networks for Embedded AI: Pruning, Projection, and Quantization](https://www.youtube.com/watch?v=7uV3-eTB5es)

**Technical Tutorials:**
- [PyTorch Pruning Tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- [MATLAB Deep Learning Onramp](https://matlabacademy.mathworks.com/details/deep-learning-onramp/deeplearning)
- [Introduction to Embedded Machine Learning (Coursera/edX)](https://www.coursera.org/learn/introduction-to-embedded-machine-learning)

**Code Examples:**
- [1D CNN for Vibration-Based Fault Diagnosis (GitHub)](https://github.com/biswajitsahoo1111/cbm_codes_open)
- [MATLAB: Rolling Element Bearing Fault Diagnosis](https://www.mathworks.com/help/predmaint/ug/rolling-element-bearing-fault-diagnosis-using-deep-learning.html)

*Feel free to explore beyond these, and share anything interesting you find with me!*

---

## 🌟 Stretch Goals

If the core milestones are completed ahead of schedule, consider exploring:

- **Embedded Code Generation:** Generate C/C++ code from the trained model using code generation tools
- **Alternative Architectures:** Experiment with LSTMs or other models and compare results
- **Anomaly Detection:** Train a new model with only "Normal" data for an anomaly detection task

---

## 🤝 How We'll Work Together

**Check-ins:** During our biweekly 60-min AI Studio Lab Section meeting block (2nd and 4th week of every month)  
**Communication:** Slack (Break Through Tech workspace)  
**Response time:** Within 48 hours on weekdays  

**Recommended Tools:**
- **Coding:** Google Colab, MATLAB Online
- **Collaboration:** GitHub, Notion
- **Virtual Meetings:** Zoom, Google Meet

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Explore the dataset** in the [data folder](data) — try loading a .mat file in Python or MATLAB
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)
4. **Pick a resource** from the list above and start familiarizing yourself with the problem domain

I'm excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech's Bridge to Studio - Session B).

---
