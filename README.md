# 🌍 Global Trade AI Dashboard

## 📌 Overview
The Global Trade AI Dashboard is an interactive data analytics and machine learning system designed to analyze and predict international trade flows. The project integrates data visualization with predictive modeling to provide meaningful insights into global trade patterns.

---

## 🎯 Objectives
- Analyze historical international trade data  
- Identify trends and country-to-country trade relationships  
- Detect anomalies and significant changes in trade flow  
- Predict future trade values using machine learning models  

---

## 🚀 Features

### 📊 Interactive Dashboard
- Dynamic filtering by Year, Reporter Country, Partner Country, and Trade Flow  
- Real-time updates of charts and insights  

### 📈 Data Analysis
- Trade value trends over time  
- Country-wise trade comparison  
- Reporter vs Partner heatmap  
- Trade distribution visualization  

### 🤖 Machine Learning Module
- Trade value prediction using:
  - Random Forest Regressor  
  - LightGBM  
- Future forecasting up to 2030  
- Actual vs Predicted comparison  

### 🧠 AI Insights
- Automated summary of trade patterns  
- Identification of dominant trade corridors  
- Anomaly detection in trade data  
- Key insights for decision-making  

### 📉 Model Evaluation
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- R² Score  
- Model comparison and best model selection  

---

## 🛠️ Technology Stack
- Python  
- Streamlit  
- Pandas, NumPy  
- Scikit-learn  
- LightGBM  
- Matplotlib, Seaborn

---


## 📂 Project Structure

.
├── app.py
├── requirements.txt
├── trade_1988_2021.csv
├── README.md


---

## 📊 Dataset
The application uses a dataset containing international trade records.

**Required Columns:**
- Year  
- ReporterName  
- PartnerName  
- TradeValue in 1000 USD  

**Optional Column:**
- TradeFlowName  

---

## ⚙️ Installation and Execution

### 1. Clone the Repository

git clone https://github.com/DKS-2020/trade-ai-dashboard.git

cd trade-ai-dashboard


### 2. Install Dependencies

pip install -r requirements.txt


### 3. Run the Application

streamlit run app.py


---

## 🎯 Applications
- Economic and trade analysis  
- Policy and strategic decision-making  
- Academic and research projects  
- Machine learning and data visualization demonstrations  

---

## 🔮 Future Scope
- Integration of real-time trade data  
- Advanced AI and deep learning models  
- Deployment on cloud platforms  
- Enhanced explainable AI features  

---

## 👨‍💻 Contributors
- Deepak Kumar Singh (Team Lead)  
- Aniket Sharma  
- Abhishek Pratap Singh  
- Atharva Deshmukh  

---
