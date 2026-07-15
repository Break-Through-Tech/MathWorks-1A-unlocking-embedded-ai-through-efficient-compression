---

> ## Challenge Advisor: Update & Finalize Your Project Overview
>
> > 💡 **These grey text instructions are just for you, the team's Challenge Advisor; please delete them once you have completed the steps below.**
>
> We've pre-populated this Challenge Project Overview page — which is what will be shared with your Break Through Tech student team in August — using the details from your submission form. You should have received an email inviting you to join this repo as a Collaborator, enabling you to add files and make edits.
> 
> In order for your project to be finalized and assigned to a team, please:
> 1. **Review all sections below** and update or expand any content as needed, making sure to address the SME Feedback in the section immediately below. Look for square brackets to find the places below that require additional inputs from you (e.g., "About [Company / Org Name]").
> 2. **Add your dataset** to the [data folder](data) in this repo.
> 3. **Close the Issue assigned to you in this repo** to let us know that you have made your edits and the overview page is ready for final review. You can do this by going to the _Issues_ tab in the top left section of the menu above, add a comment that says "CA review complete", and click the button to Close the Issue. 
>
> If you're unfamiliar with how to edit a page like this in GitHub, check out [this tutorial](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/handson/edit-readme.html) for a quick overview (start with step 2 and only edit this page), and [this guide](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/markdown.html) on how to use Markdown to compose text.
>
>
> ❌ Remember that this is a public repo. Do NOT include: Proprietary data, PII, API keys, credentials, or anything confidential.

---
## 📋 BTT Internal Evaluation Notes
*(This section is for BTT staff only — remove before sharing with students)*

| Check | Status | Notes |
|-------|--------|-------|
| Python Compatibility |🔴 | The dataset is provided exclusively in MATLAB data format (.mat), and the milestones explicitly require students to use a "Simulink model" and provide "feedback on using MathWorks tools." |
| Data Readiness | 🟢 | The dataset scale is safe (Less than or equal to 1 GB),  with zero risk of Google Colab Out-of-Memory (OOM) crashes. However, it only contains partial documentation and will require some cleaning and preprocessing before model training can begin. |
| Resource Check | 🟡 | The dataset being exclusively in MATLAB data format requires deep learning model compression (quantization/pruning) and navigating proprietary software licensing, all of which raise the difficulty outside the timeline, resource, and competency scope.|

**Student Fit Score:** 7/10  
**Technical Depth Score:** 8/10  
**Overall Recommendation:** REVISE

**Advisor Feedback Draft:**
The project presents a solid challenge and real-world value in optimizing edge models for the Fellows. However, as is the project currently relies on proprietary MATLAB/Simulink files, creating a tooling compliance block. To proceed, I recommend converting the dataset to standard open-source formats (like .csv) so fellows can execute all deep learning and compression tasks using pure Python frameworks.  

---

# Smarter Devices, Smaller Models: Unlocking Embedded AI Through Efficient Compression

**Company / Org:** MathWorks  
**Challenge Advisor:** Nayara Aguiar, ngomesde@mathworks.com  
**Program:** Break Through Tech AI Studio - Fall 2026

---

## 🏢 About MathWorks

MathWorks is a leader in technical computing and model-based design. We provide software solutions that empower engineers and scientists to create and analyze complex systems. Our tools are key in industries such as automotive, aerospace, and communications.

---

## 🎯 The Challenge

### Project Summary
In this project, you will use sensor data and deep learning to develop a smart appliance system optimized for embedded applications. You will learn how to build, train, and compress AI models while providing helpful insights about the usability of MathWorks products.

### Success Criteria
Development of a model for a smart appliance application; performance analysis including inference speed, memory footprint, and accuracy; ability to discuss deep learning model compression and trade-offs; feedback on MathWorks tools.

### Project Milestones

Use these milestones to guide your work. Your team will create a **GitHub Projects board** to track tasks within each milestone.

| Month      | Milestone                     | Key Activities                                                      |
|------------|-------------------------------|--------------------------------------------------------------------|
| **September** | Data Understanding              | Explore dataset, handle missing values, document findings          |
| **October** | Model Development              | Train baseline model, experiment with approaches, iterate         |
| **November** | Evaluation & Presentation      | Finalize model, prepare presentation, document results            |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset

**Name and Source:** Rolling Element Bearing Fault Diagnosis (from public GitHub repository)  
**Format:** MATLAB (.mat)  
**Size:** under 1gb  
**Location:** [Link to dataset or instructions for accessing it]

### Key Details
- Numerical and Time Series sensor signal data (Rolling Element Bearing Fault Diagnosis) provided in MATLAB (.mat) format from a public GitHub repository.
- [Any known limitations or preprocessing needed]
- [Link to data dictionary or documentation, if available]

---

## 🛠️ Suggested Approach

**ML Problem Type:** Classification

**Recommended Libraries:**
- Deep Learning
- Neural Networks
- MATLAB
- Simulink
- MathWorks tools
- GitHub
- Google Colab

**Evaluation Metrics:**
- Accuracy
- Inference Speed
- Memory Footprint

---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- [Link to an article or blog post about the problem domain]
- [Link to an industry report or case study]

**Technical Tutorials:**
- [Link to a free tutorial on the ML technique(s) involved]
- [Link to documentation for a key library or tool]

**Code Examples:**
- [Link to a relevant GitHub repo]
- [Link to a sample implementation or starter code]

**Other:**
- [Links to any additional resources — e.g., papers, videos, podcasts, etc.]

*Feel free to explore beyond these, and share anything interesting you find with me!*

---

## 🤝 How We'll Work Together

**Check-ins:** During our biweekly 60-min AI Studio Lab Section meeting block (2nd and 4th week of every month)  
**Communication:** Slack (Break Through Tech workspace)  
**Response time:** Within 48 hours on weekdays  

**Recommended Tools:**
- **Coding:** Google Colab
- **Collaboration:** GitHub, Notion
- **Virtual Meetings:** Zoom, Google Meet

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I'm excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech's Bridge to Studio - Session B).

---
