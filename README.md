# Breast Cancer Prediction with Logistic Regression
---

<img width="1280" height="520" alt="logistic_regression" src="https://github.com/user-attachments/assets/bba546bb-a3d4-4308-879d-89119ec2c461" /> 

---
## 📌 Introduction
In this project, I applied **Logistic Regression** to the **Breast Cancer Wisconsin dataset**.  
The dataset contains detailed measurements of cell nuclei, along with a diagnosis label:  
- **0 → Benign**  
- **1 → Malignant**

The main goal was to build and evaluate a classification model that predicts whether a tumor is malignant, based on the input features.

---

## 🎯 Objectives
1. Understand the workflow of logistic regression in practice.  
2. Perform data cleaning and preprocessing:
   - Drop useless columns (`id`, `Unnamed: 32`).  
   - Encode target variable (`diagnosis`) into binary (0/1).  
   - Standardize features using `StandardScaler`.  
3. Train and evaluate a logistic regression model.  
4. Analyze model performance with:
   - Accuracy, Precision, Recall, F1-Score  
   - Confusion Matrix  
   - ROC Curve & AUC Score  
   - Threshold tuning experiments  
5. Explain the role of the **sigmoid function** in logistic regression and how the decision threshold affects predictions.  

---

## ⚙️ Workflow
1. **Import libraries** (NumPy, Pandas, Seaborn, Matplotlib, scikit-learn).  
2. **Load dataset** (`df = pd.read_csv("dataset.csv")`).  
3. **Explore data**: shape, summary statistics, missing values.  
4. **Visualize missingness** with a heatmap.  
5. **Clean dataset**: drop irrelevant columns, encode target.  
6. **Preprocessing**: scale numeric features with `StandardScaler`.  
7. **Split dataset** into training and test sets.  
8. **Train Logistic Regression model**.  
9. **Evaluate model**:
   - Classification Report (Precision, Recall, F1).  
   - Confusion Matrix.  
   - ROC-AUC Curve.  
10. **Threshold tuning** to study precision-recall tradeoffs.  
11. **Explain sigmoid function** and decision boundary.  

---

## 📊 Results
- **Accuracy**: ~97% on the test set.  
- **ROC-AUC Score**: ~0.99.  
- Both precision and recall were very high, showing the model is well-suited for medical prediction tasks.  

Figures and reports saved in the `outputs/` folder:  
- `missing_heatmap.png`  
- `class_distribution.png`  
- `confusion_matrix.png`  
- `roc_curve.png`  
- `summary_statistics.csv`  
- `classification_report.csv`  
- `scaled_features.csv`  

---

## ✅ Conclusion
- Logistic Regression performed exceptionally well on this dataset.  
- The model achieved strong accuracy and balanced performance across precision and recall.  
- **Threshold tuning** showed how adjusting the decision rule can favor recall (catching more malignant cases) or precision (reducing false alarms), depending on the clinical context.  
- The project demonstrates the practical workflow of logistic regression, making it a solid foundation for more advanced models.  

---

## 🛠️ Tech Stack
- **Programming Language**: Python 3  
- **Data Manipulation**: Pandas, NumPy  
- **Data Visualization**: Matplotlib, Seaborn  
- **Machine Learning**: scikit-learn (Logistic Regression, preprocessing, evaluation metrics)  
- **Environment**: Jupyter Notebook  

---

## 📚 Acknowledgements & References
This project was inspired by tutorials, textbooks, and open-source resources on data science and logistic regression.

These resources helped me understand both the practical implementation of logistic regression in Python and the theoretical foundations behind regression analysis, coefficient interpretation, and model validation.

### Books and Papers Consulted
I used the following resources while working on this project, both for coding implementation and theoretical understanding of logistic regression:

1. *A Comparison of Classification/Regression Trees and Logistic Regression in Failure Models* — Irimia-Dieguez, A.I., Blanco-Oliver, A., Vazquez-Cueto, M.J. (2nd GLOBAL CONFERENCE on Business, Economics, Management and Tourism, 2014).  
2. *An Introduction to Logistic Regression: From Basic Concepts to Interpretation with Particular Attention to Nursing Domain* — Park, Hyeoun-Ae, Seoul National University, Korea.  
3. *Comparing performances of logistic regression, classification and regression tree, and neural networks for predicting coronary artery disease* — Imran Kurt, Mevlut Ture, A. Turhan Kurum.  
4. *Logistic Regression Session 1* — University of Arizona.  
5. *Logistic regression and artificial neural network classification models: a methodology review* — Stephan Dreiseitl, Lucila Ohno-Machado.  
6. *Classification with Logistic Regression* — Chad Wakamiya, Spring 2020 DATA X.  

### Suggested References (Openly Available)
- [*An Introduction to Statistical Learning* – James, Witten, Hastie, Tibshirani (Free PDF)](https://www.statlearning.com/)  
- [Scikit-Learn Documentation — Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)  
- [UCI Repository: Breast Cancer Wisconsin Dataset](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic))  

---

## ✍️ Author
Arao Macaia

AI & ML Internship Task 4 (2025)
