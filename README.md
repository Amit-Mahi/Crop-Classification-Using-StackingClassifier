# 🌱 Crop Classification using Stacking Ensemble

## 📌 Project Overview  
This project applies **machine learning ensemble methods** to classify crops based on soil and environmental features.  
We use a **Stacking Classifier** that combines multiple base learners (**SVM, Logistic Regression, KNN, CART**) with a **Random Forest meta-learner** to improve prediction accuracy.  

The model achieves **~99.5% accuracy** on the dataset.  

---

## 📂 Dataset  
The dataset contains the following features:  

- **N** → Nitrogen content (mg/kg)  
- **P** → Phosphorus content (mg/kg)  
- **K** → Potassium content (mg/kg)  
- **temperature** → Temperature (°C)  
- **humidity** → Relative Humidity (%)  
- **ph** → Soil pH value  
- **rainfall** → Rainfall (mm)  
- **label** → Crop type (target variable)  

---

## ⚙️ Project Workflow  

1. **Data Preprocessing**
   - Load dataset with pandas  
   - Handle missing values (if any)  
   - Train-test split (80-20 split)  
   - Standardization with `StandardScaler`  

2. **Base Learners (Level-0 Classifiers)**
   - Support Vector Machine (SVM)  
   - Logistic Regression (LR)  
   - K-Nearest Neighbors (KNN)  
   - Decision Tree (CART)  

3. **Meta Learner (Level-1 Classifier)**
   - Random Forest Classifier  

4. **Stacking Ensemble**
   - Combine predictions from base learners  
   - Train Random Forest on stacked predictions  
   - Evaluate performance  

5. **Evaluation Metrics**
   - Accuracy (%)  
   - Confusion Matrix  
   - ROC-AUC Score  
   - ROC Curves (One-vs-Rest strategy)  

6. **Hyperparameter Tuning**
   - Used `GridSearchCV` with `StackingClassifier`  
   - Tuned base learner & meta-learner hyperparameters  
   - Achieved best accuracy: **99.55%**  

---

## 📊 Results  

- **Best Accuracy (Test Set): ~99.55%**  
- Stacking outperformed individual models.  
- Random Forest was the best choice as meta-learner.  

---

## 📌 Installation & Usage  

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/crop-classification.git
cd crop-classification
