# Bank Forecast AI (Ollama Local Project)

An AI-powered assistant predicting **loan default risk** and **branch liquidity**, enriched with **local LLM explanations** via Ollama (Qwen 2.5 7B).


Project Title: AI-Driven Loan Default & Liquidity Forecast Assistant (Powered by Local LLM – Qwen2.5 via Ollama)

Objective
The project builds a banking intelligence assistant that combines:

1. Machine Learning models → to predict who might default on a loan and how each branch’s cash flow will behave over the next 7 days, and

2. a Local Large Language Model (Qwen2.5:7B via Ollama) → to explain these predictions and answer natural-language banking questions.

It’s designed to demonstrate how predictive analytics and generative AI can work together to make financial decisions transparent and human-readable — all running entirely on a local machine (no cloud dependency).



Two Core ML Components
1. Loan Default Prediction (XGBoost)
        --Uses customer features (age, income, loan amount, tenure, EMI history, etc.)
        --Predicts probability of default for each customer.
        --Helps a bank identify high-risk borrowers before they miss payments.

2. Liquidity Forecasting (LSTM)
            --Trains on each branch’s daily inflow/outflow transactions.
            --Forecasts cash balance for the next 7 days.
            --Useful for treasury and branch operations to plan fund allocations.


Generative AI Component (Qwen2.5 LLM)
        --The LLM is integrated through Ollama (a local model runner).
        --It’s called via ask_ollama() function using the command:
        --ollama run qwen2.5:7b --prompt "<your question>"


Used for:
Explaining model outputs in plain English
Summarizing risk and liquidity trends
Answering custom user queries like:
“Why is the liquidity of BR03 decreasing this week?”
“List possible reasons for customer defaults in the last quarter.”
So, the LLM turns raw numeric forecasts into human-interpretable business insights — mimicking a data analyst.




 Architecture Overview
+------------------------------------------------+
|              Streamlit Dashboard               |
|  • Default Risk Table                          |
|  • Liquidity Forecast Chart                    |
|  • Chat Assistant (LLM powered)                |
+-------------------▲----------------------------+
                    │
                    │ REST / Local Calls
+-------------------+----------------------------+
|        Forecast Engine (Python ML)             |
|  • XGBoost  → Default Prediction               |
|  • LSTM     → Liquidity Forecast               |
+-------------------▲----------------------------+
                    │
                    │ Prompt-based
+-------------------+----------------------------+
|       Ollama Local Model Runner                |
|  • Qwen2.5:7B LLM                              |
|  • Generates textual insights & explanations   |
+------------------------------------------------+

 Technology Stack
Layer	                    Tools Used
ML Frameworks	            XGBoost, TensorFlow (LSTM)
Data Handling	            Pandas, NumPy
App Framework	            FastAPI + Streamlit
LLM Runtime	                Ollama (Qwen2.5:7B)
Language	                Python 3.11+
Visualization	            Plotly / Streamlit Charts
Platform	                Works on both macOS & Windows


 Business Use-Case Perspective

Banking & Finance Departments use it for:
Early detection of customer defaults
        --Efficient branch-level liquidity management
        --Generating plain-language summaries for management reports

Innovation Goal:
Showcase how traditional analytics (ML) and conversational AI (LLM) together create “explainable banking intelligence.”

Expected Outcomes
Output	                                Description
 Loan Default Risk Table	            Customers ranked by default probability
 Branch Liquidity Forecast	            7-day predicted cash balances per branch
 LLM-Generated Insights	                Human-readable summary of trends and risks
 Chat Assistant	                        Natural-language Q&A on data and forecasts


 What You’ll Present

During your industry finals, you can demonstrate:

The dashboard interface — three tabs (Default, Liquidity, Chat).

How the models run locally (no external API calls).

A live Qwen2.5-based explanation (“Explain why branch BR01 liquidity dropped”).

Discussion of AI interpretability and privacy (local inference = secure).

Windows bash:

run_win.bat


Windows: bank_forecast_ai/run_win.bat

python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
ollama pull qwen2.5:7b
streamlit run dashboard\app.py


After saving all file RUN

bash
python generate_data.py
python app/forecast_engine.py
streamlit run dashboard/app.py






DOWNLAOD OLLAMA FROM HERE BELOW LINL:
https://ollama.com/download/windows

https://www.youtube.com/watch?v=Xtwrk_MGcIs LEARN AND INSTALL OLLAMA and call qwen 2.5 LLM


