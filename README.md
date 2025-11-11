# **SMS Spam Detection Using Naive Bayes (From Scratch)**

This project implements a **Spam vs. Ham SMS classification system from scratch** using **Multinomial Naive Bayes** and **Bernoulli Naive Bayes**.
All algorithms, preprocessing, cross-validation, evaluations, and visualizations are implemented manually **without using scikit-learn classifiers**.

---

## 🚀 **Project Overview**

This project builds a complete text-classification pipeline:

* Load and parse the **SMS Spam Collection Dataset**
* Text preprocessing:

  * Lowercasing
  * Punctuation removal
  * Tokenization
  * Custom stopword removal
* Manual vocabulary construction
* Implementation of:

  * Multinomial Naive Bayes
  * Bernoulli Naive Bayes
  * Laplace smoothing
  * Log-priors and log-likelihoods
* Manual **5-Fold Cross-Validation**
* Evaluation metrics:

  * Accuracy
  * Precision
  * Recall
  * F1-Score
  * Confusion Matrix
* Visualization:

  * Class distribution
  * Fold-wise metric comparison
  * Confusion matrix heatmaps (per fold & averaged)
  * Misclassified message inspection

---

## 🧠 **Implemented Models**

### **1. Multinomial Naive Bayes**

* Works with word counts
* Achieves **highest accuracy, precision, and F1-score** in this project
* Fewer false positives than Bernoulli NB

### **2. Bernoulli Naive Bayes**

* Works with binary word presence
* Higher recall (detects more actual spam)
* But produces more false positives

---

## 📂 **Project Structure**

```
CS_ML_Project_Dharmraj.ipynb
README.md
SMSSpamCollection      # Dataset file (to be placed in working directory)
```

---

## 🛠️ **Tech Used**

**Python, NumPy, Matplotlib, Seaborn, Regular Expressions**

(No ML libraries such as scikit-learn, NLTK, SpaCy, or TensorFlow were used.)

---

## 📈 **5-Fold Cross-Validation**

The notebook performs manual 5-fold CV:

* Each fold trains both NB models
* Metrics (Accuracy, Precision, Recall, F1) are computed for each fold
* Results are averaged and compared in a summary table
* Confusion matrices are plotted per fold for both models
* Misclassified messages (FP & FN) are extracted and printed for analysis

---

## 🔍 **Key Findings**

* **Multinomial NB** performed best overall:

  * ~0.98 accuracy
  * High precision & F1-score
  * Balanced confusion matrices

* **Bernoulli NB** showed:

  * High recall (~0.99)
  * But many false positives
  * Much lower precision & accuracy

* Dataset is imbalanced (more "ham" than "spam"), affecting model behavior.

---

## 🖼️ **Visualizations Included**

* Spam vs. Ham class distribution
* Accuracy/Precision/Recall/F1 comparison charts (per fold + average)
* Confusion matrix heatmaps for every fold
* Averaged confusion matrices
* Examples of false positives & false negatives

---

## ▶️ **How to Run**

1. Place the dataset file **SMSSpamCollection** in your working directory.
2. Run the notebook:

```bash
python3 CS_ML_Project_Dharmraj.ipynb
```

Or open it in **Google Colab**:

* Upload the notebook
* Upload `SMSSpamCollection`
* Run all cells

---

## 📚 **Dataset**

This project uses the **UCI SMS Spam Collection Dataset**, containing 5,574 SMS messages labeled as:

* `ham` → non-spam
* `spam` → spam


---

## 🧾 **Conclusion**

This project demonstrates:

* End-to-end text processing without external NLP libraries
* Two Naive Bayes algorithms implemented manually
* Robust experimentation using cross-validation
* Clear insights through detailed evaluation and visualization

Multinomial Naive Bayes is the more reliable model for this dataset!
