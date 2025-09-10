# 📌 E-Commerce Sales Forecasting & Product Recommendation

## 📖 Overview

This project builds a **machine learning pipeline** for an **E-commerce company** to:

1. 📈 **Predict future sales trends** using **time-series forecasting** (Facebook Prophet).
2. 🛍️ **Recommend products** using a **hybrid recommendation system** (Popularity-based + Content-based filtering).

The goal is to help businesses:

* Identify **future demand trends**.
* Recommend **high-demand products** by region, channel, and item type.
* Support **expansion strategies** across regions & sales channels.

---

## 📂 Dataset

* **Source**: <a href ="E-commerce-product_dataset.csv">Dataset</a>
* **Columns Used**:

  * `Order Date` (dd-mm-yyyy)
  * `Item Type`
  * `Region`
  * `Country`
  * `Sales Channel` (Online/Offline)
  * `Units Sold`
  * `Total Revenue`, `Total Profit`

---

## ⚙️ Methodology

### 1. **Data Preprocessing**

* Converted `Order Date` → Extracted `Year` & `Month`.
* Encoded categorical features (`Item Type`, `Region`, `Country`, `Sales Channel`).

### 2. **Forecasting Model (Future Sales)**

* Used **Facebook Prophet** to predict sales growth for the next **5 years**.
* Forecasts were done for:

  * Overall company sales
  * Item type trends
  * Region-specific growth

### 3. **Recommendation System**

* **Popularity-based**: Ranks top products by total units sold.
* **Content-based**: Uses **cosine similarity** on encoded features.
* **Hybrid System**: Combines **popularity + similarity + regional filters**.

### 4. **Evaluation**

* Visualized trends with **Matplotlib, Seaborn, Plotly**.
* Compared actual vs forecasted sales.

---

## 📊 Results

✅ **Forecasting:**

* Sales expected to grow steadily with product-specific variations.
* Seasonal & regional differences observed.

✅ **Recommendation System:**

* Suggests relevant products based on **region, channel, and similarity**.
* Hybrid approach improves over simple popularity-based methods.

---

## 🚀 Tech Stack

* **Python 3.10+**
* **Libraries**:

  * `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`
  * `scikit-learn` (LabelEncoder, Cosine Similarity)
  * `prophet` (Time-series forecasting)

---

## 📦 How to Run

```bash
# Clone repository
git clone https://github.com/your-username/ecommerce-forecast-recommender.git
cd ecommerce-forecast-recommender

# Install dependencies
pip install -r requirements.txt

# Run analysis
python main.py
```

---

## 🔮 Future Scope

* Add **collaborative filtering (Matrix Factorization / Neural Nets)**.
* Deploy as a **Streamlit/Dash web app** for interactive dashboards.
* Include **region-wise forecasting dashboards** for business expansion.
* Automate retraining with **live sales data pipelines**.

---

## 👨‍💻 Author

**Your Name**

* 📧 Email: [E-mail](mailto:vkt.vivek007@gmail.com)
* 🔗 [LinkedIn](https://www.linkedin.com/in/vivek-kumar-tiwari-9806b2299/)
* 🐙 GitHub: [your-username](https://github.com/Virvivek007/Virvivek007)


