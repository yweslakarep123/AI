import streamlit as st
import time
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import matplotlib.pyplot as plt

# Set page config for better mobile experience
st.set_page_config(layout="wide", page_title="Stock Prediction App", page_icon="📈")

# Custom CSS for better styling
st.markdown("""
<style>
    body {
        color: #FF3333;
        background-color: #000000;
    }
    .stApp {
        background-color: #000000;
    }
    .reportview-container .main .block-container {
        max-width: 1000px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF3333;
        color: white;
    }
    .stSlider>div>div>div>div {
        background-color: #FF3333;
    }
    h1, h2, h3 {
        color: #FF3333 !important;
    }
    .sidebar .sidebar-content {
        background-color: #333333;
    }
    .Widget>label {
        color: #FF3333;
    }
    .stSelectbox>div>div>div {
        background-color: #333333;
        color: #FF3333;
    }
    p {
        color: #FF3333 !important;
    }
    .stDataFrame {
        color: #FF3333;
    }
    .dataframe {
        color: #FF3333;
    }
</style>
""", unsafe_allow_html=True)

START = "2014-01-01"
TODAY = time.strftime("%Y-%m-%d")

# Title with custom styling
st.markdown("<h1 style='text-align: center; color: #FF3333;'>📈 Stock Prediction App 📈</h1>", unsafe_allow_html=True)

# Sidebar for user inputs
st.sidebar.header("User Input Parameters")

stocks = (
    "AAPL", "GOOGL", "MSFT", "AMZN", "NVDA", "META", "TSLA", "JPM", 
    "V", "WMT", "JNJ", "PG", "DIS", "NFLX", "INTC", "AMD", "BA", "GE"
)
selected_stocks = st.sidebar.selectbox("Select dataset for prediction", stocks)

n_years = st.sidebar.slider("Years of prediction", 1, 5, 2)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("About this app")
    st.markdown("<p style='color: #FF3333;'>This app predicts future stock prices using the Prophet model.</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #FF3333;'>You've selected: {selected_stocks}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #FF3333;'>Prediction period: {n_years} years</p>", unsafe_allow_html=True)

with col2:
    data_load_state = st.empty()
    data_load_state.markdown("<p style='color: #FF3333;'>Loading data...</p>", unsafe_allow_html=True)
    data = load_data(selected_stocks)
    data_load_state.markdown("<p style='color: #FF3333;'>Loading data... done!</p>", unsafe_allow_html=True)

# Display raw data in an expander
with st.expander("View Raw Data"):
    st.subheader("Raw data")
    st.dataframe(data.tail().style.set_properties(**{'color': '#FF3333', 'background-color': 'black'}))

# Plot raw data
st.subheader("Historical Data Visualization")
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open", line=dict(color="#00FFFF")))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close", line=dict(color="#FF3333")))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True, 
                      paper_bgcolor="black", plot_bgcolor="black", font=dict(color="#FF3333"))
    fig.update_xaxes(gridcolor="#333333", zerolinecolor="#333333")
    fig.update_yaxes(gridcolor="#333333", zerolinecolor="#333333")
    st.plotly_chart(fig, use_container_width=True)

plot_raw_data()

# Predict forecast with Prophet
df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

# Show and plot forecast
st.subheader("Forecast Data")
with st.expander("View forecast data"):
    st.dataframe(forecast.tail().style.set_properties(**{'color': '#FF3333', 'background-color': 'black'}))

st.subheader("Forecast Visualization")
fig1 = plot_plotly(model, forecast)
fig1.update_layout(paper_bgcolor="black", plot_bgcolor="black", font=dict(color="#FF3333"))
fig1.update_xaxes(gridcolor="#333333", zerolinecolor="#333333")
fig1.update_yaxes(gridcolor="#333333", zerolinecolor="#333333")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Forecast Components")
fig2 = model.plot_components(forecast)
plt.style.use('dark_background')
for ax in fig2.axes:
    ax.tick_params(colors='#FF3333')
    ax.xaxis.label.set_color('#FF3333')
    ax.yaxis.label.set_color('#FF3333')
    ax.title.set_color('#FF3333')
    for spine in ax.spines.values():
        spine.set_edgecolor('#FF3333')
fig2.patch.set_facecolor('black')
st.pyplot(fig2)

