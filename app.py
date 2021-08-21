# Predict ticker forecast with Prophet
import streamlit as st
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from datetime import date

# Select
st.set_page_config(layout='wide')
c1, c2 = st.columns((1,6))

tickers = ('ABEV3','ALPA4','AMER3','ASAI3','AZUL4','B3SA3','BBAS3','BBDC3','BBDC4','BBSE3','BIDI11','BIDI4','BPAC11','BPAN4','BRAP4','BRDT3','BRFS3','BRKM5','BRML3','BTOW3','CASH3','CCRO3','CIEL3','CMIG4','COGN3','CRFB3','CSAN3','CSNA3','CVCB3','CYRE3','ELET3','ELET6','EMBR3','ENEV3','EQTL3','GGBR4','GNDI3','GOAU4','GOLL4','HAPV3','HYPE3','IGTA3','INTB3','IRBR3','ITSA4','ITUB4','JBSS3','KLBN11','LAME4','LCAM3','LREN3','LWSA3','MGLU3','MRFG3','MULT3','NTCO3','PETR3','PETR4','POSI3','PRIO3','RADL3','RAIL3','RDOR3','RENT3','SANB11','SBSP3','SULA11','SUZB3','TAEE11','TOTS3','UGPA3','USIM5','VALE3','VIVT3','VVAR3','WEGE3')
ticker = c1.selectbox('Ticker', tickers)

start = c1.slider('Start', 2010, 2020, value=2015)
TODAY = date.today().strftime('%Y-%m-%d')

period = c1.slider('Years', 1, 5, value=2)

# Data
@st.cache
def load(ticker):
    data = yf.download(ticker+'.SA', str(start)+'-01-01', TODAY)
    data.reset_index(inplace=True)
    return data

data = load(ticker)

# Prophet
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={'Date': 'ds', 'Close': 'y'})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period*365)
forecast = m.predict(future)

# Plot
fig1 = plot_plotly(m, forecast)
fig1.update_layout(title=f'{ticker} from {start} forecast for {period} years', yaxis_title='Close', xaxis_title='Date', plot_bgcolor='lightgray')
c2.plotly_chart(fig1)
