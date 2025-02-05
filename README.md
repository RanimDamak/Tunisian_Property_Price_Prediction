# Tunisian Property Price Prediction

## Overview
This project aims to predict property prices in Tunisia using machine learning techniques. The dataset contains various property attributes such as type, category, city, room count, bathroom count, and size. The model is trained using a Random Forest Regressor with hyperparameter tuning through GridSearchCV.

## Dataset
- **File:** `Property Prices in Tunisia.csv`
- **Features:**
  - `type`: Property type (e.g., "À Vendre" (For Sale), "À Louer" (For Rent))
  - `category`: Type of property (e.g., apartment, villa)
  - `city`: Location of the property
  - `room_count`: Number of rooms
  - `bathroom_count`: Number of bathrooms
  - `size`: Size of the property in square meters
  - `price`: Property price (target variable)

## Steps in the Project
1. **Exploratory Data Analysis (EDA)**:
   - Checking dataset shape, missing values, and duplicates
   - Visualizing distributions of features
   - Handling missing values and outliers

2. **Data Preprocessing**:
   - Replacing `-1` values with `NaN`
   - Removing duplicates
   - Encoding categorical variables using `OrdinalEncoder`
   - Standardizing numerical variables with `StandardScaler`

3. **Feature Engineering**:
   - Creating a new feature `log_price` (log-transformed price for better predictions)
   - Removing extreme outliers using the Z-score method
   
4. **Model Training**:
   - Splitting data into training and testing sets (`train_test_split`)
   - Building a preprocessing pipeline with `ColumnTransformer`
   - Training a `RandomForestRegressor`
   - Hyperparameter tuning using `GridSearchCV`

5. **Evaluation**:
   - Measuring model performance using:
     - Mean Absolute Error (MAE)
     - Root Mean Squared Error (RMSE)
     - R-squared score
   - Visualizing predictions using scatter plots

## Installation & Usage
### Prerequisites
Ensure you have Python installed along with the required libraries:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

### Running the Notebook
1. Clone the repository or download the files:
   ```bash
   git clone https://github.com/your-repo/tunisian-property-price-prediction.git
   cd tunisian-property-price-prediction
   ```
2. Open `Tunisian_Property_Price_Prediction.ipynb` using Jupyter Notebook or Google Colab.
3. Run all cells to execute the data analysis, model training, and evaluation.

## Results
- The best model was selected using `GridSearchCV`.
- The trained model provides a good approximation of property prices with reasonable error margins.
- Feature importance analysis can be performed to understand key predictors of property prices.

## Future Improvements
- Experiment with other regression models like `XGBoost` and `Gradient Boosting`.
- Fine-tune hyperparameters further for better performance.
- Explore more feature engineering techniques.
