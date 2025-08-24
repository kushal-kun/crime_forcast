# Crime Forecasting with Machine Learning

This repository contains a machine learning project that analyzes and forecasts **crime categories** using historical incident records.  
The project was developed as part of a Kaggle competition, focusing on **feature engineering, exploratory analysis, and predictive modeling**.

## Project Workflow

1. **Data Exploration & Cleaning**
   - Removed redundant columns (`Area_Name`, `Weapon_Description`, `Premise_Description`)
   - Handled missing values in victim demographics (`Sex`, `Descent`)
   - Corrected invalid entries (negative victim ages → set to 0)

2. **Feature Engineering**
   - Extracted **month** and **day** from reported/occurred dates  
   - One-hot encoded categorical features  
   - Applied **TF-IDF Vectorizer** on the *Modus Operandi* column  
   - Scaled numerical features with **StandardScaler**

3. **Exploratory Data Analysis (EDA)**
   - Distribution of crimes across months/days  
   - Victim demographic trends (age, gender, descent)  
   - Area-based analysis of incidents

4. **Modeling**
   - Applied Linear Discriminant Analysis (LDA) as the primary classification method
   - Applied classification algorithms from **scikit-learn**  
   - Evaluated predictive performance on test data

---

## Key Insights

- LDA performed well in separating crime categories when combined with TF-IDF text features and scaled numeric data.
- Crimes are **fairly evenly distributed across months**, except the 31st (naturally fewer incidents).  
- **Male victims** are slightly more common than female victims, though the gap isn’t large.  
- Negative/invalid ages in the dataset required correction.  
- **Geographical columns** (Area_ID vs. Area_Name) were redundant and simplified for cleaner modeling.  
- Text analysis of *Modus Operandi* provided useful predictive power when vectorized.  

---

## Technologies Used

- **Python 3**  
- **NumPy, Pandas** → Data cleaning & manipulation  
- **Matplotlib, Seaborn** → Visualization  
- **Scikit-learn** → ML models, preprocessing, TF-IDF, StandardScaler  
- **Jupyter Notebook** → Interactive development  

---

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/kushal-kun/crime_forcast.git
   cd crime_forcast
