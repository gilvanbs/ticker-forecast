# Predict ticker forecast with Prophet
import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

st.set_page_config(layout='wide')
c1, c2 = st.columns((1,5))

starts = ('2020-01-01','2019-01-01','2018-01-01','2017-01-01','2016-01-01','2015-01-01')
start = c1.selectbox('Start', start, svalue='2019-01-01')
TODAY = date.today().strftime('%Y-%m-%d')

stocks = ('ABEV3.SA','ALPA4.SA','AMER3.SA','ASAI3.SA','AZUL4.SA','B3SA3.SA','BBAS3.SA','BBDC3.SA','BBDC4.SA','BBSE3.SA','BIDI11.SA','BIDI4.SA','BPAC11.SA','BPAN4.SA','BRAP4.SA','BRDT3.SA','BRFS3.SA','BRKM5.SA','BRML3.SA','BTOW3.SA','CASH3.SA','CCRO3.SA','CIEL3.SA','CMIG4.SA','COGN3.SA','CRFB3.SA','CSAN3.SA','CSNA3.SA','CVCB3.SA','CYRE3.SA','ELET3.SA','ELET6.SA','EMBR3.SA','ENEV3.SA','EQTL3.SA','GGBR4.SA','GNDI3.SA','GOAU4.SA','GOLL4.SA','HAPV3.SA','HYPE3.SA','IGTA3.SA','INTB3.SA','IRBR3.SA','ITSA4.SA','ITUB4.SA','JBSS3.SA','KLBN11.SA','LAME4.SA','LCAM3.SA','LREN3.SA','LWSA3.SA','MGLU3.SA','MRFG3.SA','MULT3.SA','NTCO3.SA','PETR3.SA','PETR4.SA','POSI3.SA','PRIO3.SA','RADL3.SA','RAIL3.SA','RDOR3.SA','RENT3.SA','SANB11.SA','SBSP3.SA','SULA11.SA','SUZB3.SA','TAEE11.SA','TOTS3.SA','UGPA3.SA','USIM5.SA','VALE3.SA','VIVT3.SA','VVAR3.SA','WEGE3.SA')
selected_stock = c1.selectbox('Ticker', stocks, svalue='PETR4.SA')

n_years = c1.slider('Years', 3, 5)
period = n_years * 365

@st.cache
def load_data(ticker):
    data = yf.download(ticker, start, TODAY)
    data.reset_index(inplace=True)
    return data

data = load_data(selected_stock)

# Predict forecast with Prophet
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={'Date': 'ds', 'Close': 'y'})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
fig1 = plot_plotly(m, forecast)
fig1.update_layout(title=f'Forecast for {n_years} years', yaxis_title='Close', xaxis_title='Date', plot_bgcolor='lightgray')
c2.plotly_chart(fig1)
