# 💳 Credit Card Customer Segmentation

Unsupervised machine learning project using **KMeans clustering** to segment credit card customers for AllLife Bank.  
The project delivers insights framed in a **business analysis workflow**: Context → Objective → Data → Actions → Observations → Results.

🔗 [Live Streamlit App](https://credit-card-customer-clustering-8nsa98ruztdgc6wktz3hfa.streamlit.app/)

## 📌 Context
AllLife Bank aims to:
- Improve **credit card penetration** in the market.  
- Address **negative perceptions of customer service**.  

Marketing and Operations required data-driven insights to design **personalized campaigns** and **optimize service delivery**.

## 🎯 Objective
Segment customers based on:
- **Credit capacity** → average credit limit, number of cards  
- **Interaction patterns** → branch visits, online visits, calls  

The segmentation helps in:
- Designing **targeted marketing** campaigns  
- Reducing **service costs** via channel optimization  

## ⚙️ Actions
1. **Data Cleaning** → Removed IDs, duplicates, standardized numeric fields  
2. **Exploratory Analysis** → Histograms, boxplots, correlation heatmap  
3. **Clustering** → KMeans (K=3), validated with elbow method  
4. **Comparison** → Gaussian Mixture Models (similar clusters)  

## 📂 About the Data
**660 customer records** with the following variables:

- `Avg_Credit_Limit`  
- `Total_Credit_Cards`  
- `Total_visits_bank`  
- `Total_visits_online`  
- `Total_calls_made`  

After cleaning → **649 unique records** across **5 numeric features** were retained.

## 🔍 Observations
- **Credit Limits** are **right-skewed**  
  Median = **$18,000**, Mean = **$34,878**, Std. Dev. = **$37,813**  
- Half of customers own **3–6 cards**  
- Most interactions occur via **one dominant channel** (branch, online, or phone)  
- **Credit limit** positively correlated with **cards & online usage**, negatively with **calls & branch visits**  

## 📊 Results

### Cluster Profiles (K=3)

| Cluster | Profile | Recommended Business Action |
|---------|---------|--------------------|
| **0** | Low credit limit, very few/no cards, rely mainly on **phone support** | Enhance self-service options (IVR, chatbots) to reduce call-center load |
| **1** | Average credit limit and card ownership, prefer **branch visits** | Transition branch-heavy customers to **digital channels** |
| **2** | High credit limit, many cards, heavy **online banking usage** | Target high-value customers with **premium digital products** and loyalty programs |

### 💡 Business Impact

The segmentation exercise provides clear and actionable insights for AllLife Bank:

**Cost Optimization in Service Channels**: By identifying phone-dependent customers (Cluster 0), the bank can encourage adoption of low-cost self-service tools (IVR, chatbots, mobile app support). This reduces call center burden and operational costs.

**Digital Adoption Strategy**: Branch-reliant customers (Cluster 1) represent an opportunity to migrate toward digital banking. Educational campaigns, incentives for online transactions, and guided onboarding can shift usage away from high-cost branch visits.

**Revenue Growth from High-Value Segments**: Customers in Cluster 2 (high credit limit, many cards, digital-first) are the bank’s premium audience. They can be targeted with exclusive digital products, loyalty programs, and cross-sell opportunities (e.g., personal loans, wealth management).

**Improved Customer Experience**: By aligning service channels with customer preferences, the bank can improve satisfaction. Each cluster receives a tailored approach rather than a one-size-fits-all model.

**Strategic Marketing Allocation**: Marketing resources can be prioritized based on segment profitability and scalability. For example, premium clusters receive retention campaigns, while low-limit customers receive acquisition or upgrade offers.

👉 In short, the analysis helps AllLife Bank reduce servicing costs, increase product uptake, and improve customer satisfaction by treating each segment according to its value and preferred interaction style.

## 🛠️ Tools
- **Python** → pandas, NumPy, matplotlib, seaborn, scikit-learn  
- **Streamlit** → interactive dashboard version  


