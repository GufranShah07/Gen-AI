## Malaria Parasite Detection in Blood Smear Images using CNN and Transfer Learning
[](https://colab.research.google.com/github/gufranshah/Generative-AI-Lab/blob/main/Malaria_Parasite_Detection.ipynb)
This repository contains the implementation of a Generative AI Lab Assignment focused on reproducing, on a Colab-friendly scale, the methodology of the landmark research paper by Rajaraman et al. (2018): Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images.
## 👥 Contributors

* Student Name: GUFRAN SHAH (Student ID: 202401110014)
* Group Members: RAGAV SHARMA, PARTH KITCHLOO, SANDEEP KOTWAL
* Course: GENERATIVE AI (Submission Date: 20-08-2026)

------------------------------
## 📌 Project Objective

   1. Study and benchmark the methodology used in the reference paper for classifying single red blood cell images as Parasitized or Uninfected.
   2. Build a Custom CNN architecture from scratch to serve as a performance baseline.
   3. Apply Transfer Learning using an ImageNet pre-trained, lightweight MobileNetV2 model.
   4. Fine-tune and optimize the architectures using data augmentation and hyperparameter tuning.
   5. Evaluate and compare performance across models with reference metrics.

------------------------------
## 📊 Dataset Metadata

* Name: NIH (National Library of Medicine) Malaria Cell Images Dataset
* Source Archive: Official NIH/NLM Malaria Archive
* Volume: 27,558 single red blood cell images
* Sub-sampling strategy: Stratified, balanced subset (SAMPLE_SIZE = 4000 images; 2,000 Parasitized, 2,000 Uninfected) to avoid runtime bottlenecks.
* Input Resolution: Resized to 128 × 128 × 3 pixels using bilinear interpolation.
* Data Splits: Stratified train / validation / test splits with a 70% / 15% / 15% ratio.

------------------------------
## 🛠️ Tech Stack & Dependencies

* Core Framework: TensorFlow 2.x, Keras
* Image Processing: Pillow (PIL)
* Metrics & Analytics: Scikit-Learn
* Visualizations: Matplotlib, Seaborn

Install the non-standard libraries before execution:

pip install -U pillow scikit-learn seaborn

------------------------------
## ⚙️ Architecture & Pipeline Overview

[Input Layer] -> [Data Augmentation] -> [Preprocessing/Normalization] 
       |
       +---> Pipeline A: Custom CNN (Train from scratch) 
       |
       +---> Pipeline B: MobileNetV2 Base -> [Frozen Feature Extraction] -> [Fine-tuning top layers]

## 1. Data Augmentation
An on-the-fly augmentation step is built directly inside the Keras functional sequential container to avoid overfitting:

* Horizontal and Vertical flips [5]
* Random rotation up to 15% [5]
* Random zoom up to 10% [5]
* Random contrast alterations up to 10% [5]

## 2. Custom CNN From Scratch
Constructed with four structural convolutional layers (Conv2D with relu activations) interleaved with max-pooling structures, flattening to a dense classification engine with a dropout rate of 0.5 and a final sigmoid node.
## 3. MobileNetV2 Transfer Learning

* Stage 1 (Feature Extraction): MobileNetV2 frozen with standard ImageNet weights. A custom global average pooling network appended with an output sigmoid block is trained at a learning rate of 1 × 10⁻³.
* Stage 2 (Fine-Tuning): Layer weights from index 100 and above are unfrozen and optimized using a minimal learning rate (1 × 10⁻⁵) to adjust the abstract feature representations without wiping the pre-trained neural memory.

------------------------------
## 📈 Performance Summary
The performance indicators obtained from evaluation on the held-out test data subset are summarized below:

| Model Architecture | Test Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Custom CNN (From Scratch) | 95.33% | 0.9564 | 0.9500 | 0.9532 |
| MobileNetV2 (Fine-Tuned) | 80.67% | 0.7266 | 0.9833 | 0.8357 |

## Key Findings

* Custom CNN from Scratch demonstrated superior, well-balanced predictive accuracy on this dataset scale (95.33%).
* MobileNetV2 Transfer Learning yielded exceptionally high recall (98.33%) for the target condition but fell short in overall precision due to its heavier architectural variance with natural non-medical ImageNet parameters [14].

------------------------------
## 🔍 Execution & Usage

   1. Open the Jupyter Notebook environment or launch via Google Colab using the badge above.
   2. Run the initialization blocks to verify your GPU settings (tf.config.list_physical_devices('GPU')).
   3. Step through the data ingestion sequence. The script automatically down-samples and caches the NIH repository.
   4. Run the validation checks and plot history functions to display accuracy and loss curves.

------------------------------
## 💡 Limitations and Future Enhancements

* Dataset Scale: This experiment handles a stratified fraction (4,000 items) of the total dataset. Scalability checks on the complete corpus (27,558 samples) are needed.
* Data Leakage & Cross-Validation: Random selection partitions the dataset here. Implementing patient-level cross-validation (as detailed in the paper) will eliminate feature correlations within identical host samples.
* Interpretability: Adding Grad-CAM mapping would render the inner layers audible, highlighting the localized cellular anomalies driving predictions.

------------------------------
License: Distributed under the MIT License. Feel free to use and adapt this code for scientific and educational purposes.

