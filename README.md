# Smart Financial Intelligence Dashboard

An interactive application for analyzing and forecasting financial liquidity (Cash-Flow) using advanced econometric models and stochastic simulations. This project integrates **Computer Science** (Python, Streamlit) and **Econometrics** (Time-series modeling, Monte Carlo methods).

## Key Features

- **Time-Series Forecasting:** Utilizes the **Holt-Winters Exponential Smoothing** model to predict revenue while accounting for seasonality.
- **Monte Carlo Simulation:** Generates alternative market scenarios (200 simulations) to assess financial risk using stochastic methods.
- **Runway & Burn Rate Analysis:** Automatically calculates liquidity ratios and the estimated survival period for the business.
- **Dynamic PDF Reporting:** Generate professional business reports including integrated charts with a single click.
- **Interactive Visualization:** A comprehensive dashboard built with the Plotly library, featuring full Dark Mode support.
- **AI-Powered Insights:** Integrated with the Gemini 2.0 Flash model to generate concise CFO-style executive summaries for the board.

## Technologies

- **Language:** Python 3.9+
- **Frontend/App Framework:** Streamlit
- **Data Analysis:** Pandas, NumPy
- **Statistical Models:** Statsmodels (Holt-Winters, Seasonal Decompose)
- **Visualization:** Plotly
- **Reporting:** FPDF, Kaleido

## Optimization & Stability

The application has been optimized to handle Gemini API rate limits. It utilizes a `st.session_state` mechanism and a manual trigger button for AI analysis, preventing **429 Too Many Requests** errors during dynamic "What-If" simulation adjustments.

## Installation & Setup

- **Clone the repository:**
-   git clone <https://github.com/Huskyel/Financial-Intelligence-Dashboard.git>
- **Install the required dependencies:**
- pip install -r requirements.txt
- **Configure your API key in the .streamlit/secrets.toml file:**
- GEMINI_API_KEY = "YOUR_API_KEY_HERE"
- **Run the application:**
- streamlit run app.py