# FORECASTING OF MALAYSIA PETROL PRICES AND FACTOR INFLUENCING IT

This project was conducted as part of my Master’s degree dissertation in Data Science, focusing on the challenges of forecasting petrol prices in Malaysia—a market influenced by various macroeconomic factors. 
Traditional statistical methods, while effective for short-term predictions, struggle to capture complex, non-linear relationships and sudden market fluctuations.
To address these limitations, this study employs Vector Autoregression (VAR), Long Short-Term Memory (LSTM) models, and a hybrid LSTM-GRU approach to improve forecasting accuracy.
Key macroeconomic indicators, including international crude oil prices, gold prices, exchange rates, overnight interest rates, and the consumer price index (CPI), are analyzed using Johansen's Cointegration Test and the Granger Causality Test to identify significant long-term relationships and causal impacts on petrol prices.
A comparative analysis of the models demonstrates that the hybrid LSTM-GRU model outperforms others, achieving lower error rates and higher accuracy. 

To enhance usability, an interactive dashboard is developed, providing stakeholders with intuitive access to petrol price trends, external factor influences, actual vs. predicted values, and future forecasts.

This project not only advances the understanding of petrol price dynamics in Malaysia but also offers a robust forecasting tool for decision-makers in the energy sector.

## Data Preparation 💜

Before diving into the data preprocessing methods, you may need to collect the necessary data for this project from multiple websites.

1. https://data.gov.my/data-catalogue/fuelprice for petrol prices
2. https://www.bnm.gov.my/rates-statistics for exchange rates, kijang emas prices, and overnight interest rates
3. https://open.dosm.gov.my/publications/cpi_2025-02 for malaysia CPI
4. https://www.kaggle.com/datasets/sc231997/crude-oil-price for WTI crude oit prices

Given that the data originates from diverse sources, the challenge involves handling six distinct data files. Each data file will be meticulously cleaned to address any discrepancies, inconsistencies, or missing values that may have arisen during extraction from their respective sources. 
This cleansing process involves techniques such as imputation for missing data points, standardization of formats, and the removal of redundant or irrelevant information. You may find out more of this step in my python file.

![image](https://github.com/user-attachments/assets/84094d9a-0fa2-41b6-b0cc-955cd6590c09)

The data preparation workflow is a crucial phase in this project, ensuring data integrity and readiness for analysis. The workflow includes:

1. Data Collection & Extraction: Gathering macroeconomic variables from multiple sources based on the literature review.

2. Data Preprocessing: Handling missing values, resampling for temporal consistency, merging datasets, and normalizing data for uniformity.

3. Exploratory Data Analysis (EDA): Identifying trends, seasonality, and cyclic patterns in time series data.

4. Statistical Tests:

Granger Causality Test: Determines whether independent variables influence petrol prices.

Johansen’s Cointegration Test: Identifies long-term relationships between variables.

5. Stationarity Check: Ensuring data stability for time series modeling through transformations if needed.

This structured workflow lays the foundation for accurate forecasting and causal analysis. 🚀

### Important! Duplicated Value
![image](https://github.com/user-attachments/assets/235c2c76-24c5-4457-8e8f-560e5293cbf3)

Based on the above figure, the observation that all datasets, except for the CPI dataset, contain duplicated values is intriguing and can be attributed to the frequency of data measurement. 
While the CPI dataset is structured on a monthly basis, the others are recorded daily, making it entirely possible for them to have repeated values, particularly when market conditions remain constant over consecutive days. 
In such cases, maintaining these duplicated values can indeed be beneficial, as it allows us to have a more extensive dataset with a higher volume of data points.

### Data Resampling
In the context of this project, a significant data preprocessing step involved downsampling all datasets, except for the Consumer Price Index (CPI) dataset, from their original daily frequency to a monthly frequency. 
This transformation was implemented to align the datasets with the CPI data, which is recorded monthly. 
By harmonizing the temporal granularity across all datasets, compatibility is ensured, allowing for effective analysis together and facilitating the exploration of relationships and patterns over consistent monthly intervals. 
The down-sampling process, which converts daily data to monthly data, was achieved using the Pandas resample() function with the 'M' parameter, representing monthly frequency.

### Missing Value After Resampling
Upon resampling the dataset, it was discovered that there is no data in the petrol prices dataset for September 2018. The discovery of missing data for September 2018 in the petrol prices dataset underscores the importance of data completeness in time series analysis. 
To address this gap and maintain the continuity of the time series, an interpolation method has been applied. Interpolation involves estimating missing values by considering the neighbouring data points. 
It is a suitable method to be used in this project as we are dealing with time series data.

## Exploratory Data Analysis (EDA) 🧡
The most crucial part in any data science project is EDA. Analyzing trends and patterns within both dependent variables, including RON 95, RON 97, and Diesel, as well as independent variables such as WTI Crude Oil Prices, Ringgit Malaysia/US Dollar Exchange Rate, Selling and Buying Gold Prices, Overnight Interest Rate, and Consumer Price Index (CPI), forms the core of our Exploratory Data Analysis (EDA). 
Through conducting this comprehensive analysis, the aim is to uncover significant insights into how these variables have evolved over time and discern any recurring patterns or correlations among them.

### Fuel prices over time graph
![image](https://github.com/user-attachments/assets/a7ae3d0b-6898-4cff-91b2-8bb42e26b69f)

These fuel prices exhibit notable fluctuations over the years, characterized by multiple peaks and troughs, signifying periods of both price increases and decreases. Notably, RON 97 consistently maintains the highest price level throughout the entire time frame, establishing itself as the premium fuel option. Conversely, RON 95 generally occupies the lowest price bracket, except for a brief period between early 2018 and a few months before 2019 when its price converges with that of Diesel.
During this convergence, RON 95 and Diesel prices appear to be more closely aligned, indicating a period of relative price stability between these two fuels compared to RON 97. The data also reveals a significant downturn in fuel prices around early 2020, coinciding with the onset of the COVID-19 pandemic. 
This suggests a potential correlation between the pandemic's impact on global oil prices and demand and the observed decrease in fuel prices. However, a recovery in fuel prices becomes evident from 2021 onwards. Particularly noteworthy is the discernible upward trend in RON 97 prices, notably from early 2021 to mid-2022. The most recent data suggests a decline in RON 97 prices, while RON 95 and Diesel prices appear to be stabilizing after a period of decrease. In sum, these fuel price trends reflect a complex interplay of market dynamics, global events, and consumer preferences, with RON 97 consistently maintaining its position as the costliest fuel option throughout the analyzed time period.

### Dependent Variable Decomposition
Dependent variable decomposition involves breaking down the dependent variable into its constituent parts to gain insights into the underlying factors driving the observed outcomes.
In this analysis, the dependent variables of interest are the prices of Ron 95, Ron 97, and diesel.

![image](https://github.com/user-attachments/assets/fc56e12c-c457-4247-a26d-1c06c15745fe)

Key insights from the decomposition of RON 95 analysis include:

1. Trend Component: A declining trend from 2017 to early 2020, with prices dropping below 1.8. Around 2021, the trend reversed and gradually increased before stabilizing in 2023.
2. Seasonal Component: A consistent 12-month cycle, likely influenced by seasonal driving habits, holidays, and economic activity. The amplitude of fluctuations remains stable over time.
3. Residuals: Random fluctuations centered around zero, indicating that the trend and seasonal components effectively capture the systematic patterns in the data.

This decomposition highlights a clear seasonal pattern and a significant trend shift around 2021, providing valuable insights for forecasting and decision-making.

![image](https://github.com/user-attachments/assets/45b46a54-2d26-488c-b880-f6b55e33caa1)

The decomposition of RON97 analysis reveals:

1. Trend Component: Initially stable, prices began to decline in early 2020, reaching a low below 2.0, followed by a gradual upward trend over time.
2. Seasonal Component: A consistent 12-month cycle with stable frequency and amplitude, indicating a recurring seasonal effect.
3. Residuals: Small in magnitude, suggesting that the trend and seasonal components effectively capture most variations in the data.

Overall, the analysis highlights a slight upward trend in RON 97 prices and a stable seasonal pattern, offering insights into long-term price movements.

![image](https://github.com/user-attachments/assets/a5146ca0-3a93-44d0-bfb6-a97a0d44adce)

The decomposition analysis of Diesel reveals:

1. Trend Component: Relatively stable from 2017 to 2019, followed by a decline in mid-2020. Prices rebounded in 2021 and continued rising before stabilizing.
2. Seasonal Component: Similar 12-month cycle observed in RON 95 and RON 97, indicating shared seasonal influences.
3. Comparison with RON 95 & RON 97: Diesel and RON 95 exhibit similar long-term trends, whereas RON 97 shows a continuous upward trend after 2020.

The EDA conducted is crucial in selecting the appropriate machine learning models for this project.

## Feature Selection 💛
Initially, several macroeconomic variables were selected for this project based on a literature review. 
However, to determine whether these independent variables have a significant relationship with petrol prices, I conducted the Granger Causality Test and Johansen's Cointegration Test.

### Granger Causality Test
![image](https://github.com/user-attachments/assets/cbdc4ecf-f406-4058-b7ce-a08c142e0608)

Based on the result, For RON 95, RON 97, and diesel, the Granger causality test results reveal that the p-values for selling and buying gold prices, overnight interest rate, and CPI are all less than the conventional significance level of 0.05. 
This suggests that there is strong statistical evidence to support the existence of Granger causality between these economic indicators and the fluctuations in fuel prices. 
In other words, changes in selling and buying gold prices, overnight interest rates, and CPI can be considered as predictors or drivers of changes in the prices of RON 95, RON 97, and diesel fuels.

### Johansen's Cointegration Test
![image](https://github.com/user-attachments/assets/98d681e6-27ff-43fa-8e2a-6dc24e0b67f9)

The majority of the variables, including RON 95, RON 97, Diesel, WTI crude oil prices, the exchange rate, selling and buying gold prices, and overnight interest rates, showed statistically significant evidence of cointegration. 
This indicates that there are long-term relationships among these variables, and changes in one variable are likely to impact the others in the long run.

After considering the results of both the Granger Causality Test and Johansen's Cointegration Test, the approach is to focus on variables that not only demonstrate Granger causal relationships with the fuel prices but also exhibit cointegration with other relevant variables. 
Also, considering the size of the data and model complexity, it is important to reduce the number of independent variables. 
As a result of this dual criteria, the final set of independent variables to be included in the analysis will consist solely of selling and buying gold prices and overnight interest rates. 
These variables have demonstrated both causal influence over the fuel prices of interest and long-term relationships with other key factors, making them the most suitable candidates for inclusion in our forecasting model.

## Model Development & Evaluation 💙
![image](https://github.com/user-attachments/assets/0af7e1b0-2ab8-429a-ac79-30a723868bd6)

![image](https://github.com/user-attachments/assets/e0ba1e73-0ee3-4809-91fb-54166239bb34)

![image](https://github.com/user-attachments/assets/692d21e9-b6ed-4b5e-95fe-4643326bcf2b)

The flowcharts illustrate the development processes for the VAR, LSTM, and hybrid LSTM-GRU models used to forecast petrol prices. 
The VAR model follows a structured approach, including data splitting, logarithmic transformation, model initialization, training, Durbin-Watson statistic calculation, prediction, performance evaluation, and future price forecasting, ensuring the integrity of predictions for informed decision-making. 
The LSTM model begins with data scaling and splitting, followed by model compilation, training, and initial predictions.
Hyperparameter tuning optimizes performance, and key evaluation metrics such as RMSE, MAE, and MAPE are used to assess accuracy. 
Finally, a recursive prediction function forecasts future petrol prices, with visualizations to validate its predictive capability. 
The hybrid LSTM-GRU model follows a similar workflow but integrates both LSTM and GRU layers, along with Dropout and Dense layers, to improve learning. 
Hyperparameter tuning further enhances accuracy before employing a recursive prediction function for future forecasting and visualization. 
These structured methodologies ensure robust and reliable petrol price predictions. 🚀

## Dashboard Development 🤎
The petrol prices dashboard provides a comprehensive overview of historical and future trends in petrol prices, as well as the impact of external factors on these prices. 
The dashboard is divided into several pages, each focusing on different aspects of petrol price analysis.

![image](https://github.com/user-attachments/assets/8324082f-2707-436e-a799-9e07316fcad5)

Page 1: Petrol Prices – Displays average prices of RON 95, RON 97, and Diesel, with line charts visualizing price trends over time.

![image](https://github.com/user-attachments/assets/fc43f050-05e0-4b89-81fe-61814216a3ce)

Page 2: External Factor Prices – Highlights the influence of gold prices and overnight interest rates on petrol prices, featuring trend charts for buying/selling gold prices and interest rates.

![image](https://github.com/user-attachments/assets/520d433c-1448-4a46-ba74-e19294cebe41)
![image](https://github.com/user-attachments/assets/7c5d8a32-c93c-42ec-8f91-8362cc21051c)
![image](https://github.com/user-attachments/assets/9438b336-1c24-498d-9331-80591f4edbac)

Page 3: The "Actual vs Predicted Value" section of the dashboard provides an in-depth comparison of actual petrol prices against predicted values using VAR, LSTM, and Hybrid models. 
Divided into sections for RON 95, RON 97, and Diesel, users can toggle between models to evaluate their accuracy and performance across different training data proportions.

![image](https://github.com/user-attachments/assets/c1e51fbf-bb14-4280-bb8e-73d6d2c46c20)

Page 4: The "Forecasting" page provides a detailed analysis of future petrol price trends for RON 95, RON 97, and Diesel using various models and training data proportions. 
Users can navigate between FORECASTING 0.6, 0.7, and 0.8 to explore predictions based on different training data sizes.

### Feel free to see the actual dashboard I've included in the repository!! 🤍
