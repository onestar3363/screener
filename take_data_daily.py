import streamlit as st
import pandas as pd
import sqlalchemy
import ta
import numpy as np
import yfinance as yf
import sqlalchemy
import ccxt
import time
import pandas_ta as pa
import os
import plotly
import plotly.graph_objs as go 
import base64

st.set_page_config(layout="wide")
st.title('Screener')
start = time.perf_counter()
@st.cache(suppress_st_warning=True)
def getdata():
    if os.path.exists("günlük.db"):
        os.remove("günlük.db")
    elif os.path.exists("haftalik.db"):
        os.remove("haftalik.db")
    exchange=ccxt.currencycom()
    markets= exchange.load_markets()    
    symbols1=pd.read_csv('csymbols.csv',header=None)
    symbols=symbols1.iloc[:,0].to_list()
    index = 0
    fullnames=symbols1.iloc[:,1].to_list()
    engine=sqlalchemy.create_engine('sqlite:///günlük.db')
    enginew=sqlalchemy.create_engine('sqlite:///haftalik.db')
    with st.empty():
        for ticker,fullname in zip(symbols,fullnames):
            index += 1
            try:
                data2 = exchange.fetch_ohlcv(ticker, timeframe='1d',limit=250) #since=exchange.parse8601('2022-02-13T00:00:00Z'))
                data3= exchange.fetch_ohlcv(ticker, timeframe='1w',limit=250)
                st.write(f"⏳ {index,ticker} downloaded")
            except Exception as e:
                print(e)
            else:
                header = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                dfc = pd.DataFrame(data2, columns=header)
                dfc['Date'] = pd.to_datetime(dfc['Date'],unit='ms')
                dfc['Date'] = dfc['Date'].dt.strftime('%d-%m-%Y')
                dfc.to_sql(fullname,engine, if_exists='replace')
                dfc2 = pd.DataFrame(data3, columns=header)
                dfc2['Date'] = pd.to_datetime(dfc2['Date'],unit='ms')
                dfc2['Date'] = dfc2['Date'].dt.strftime('%d-%m-%Y')
                dfc2.to_sql(fullname,enginew, if_exists='replace')

        index += 1
        bsymbols1=pd.read_csv('bsymbols.csv',header=None)
        bsymbols=bsymbols1.iloc[:,0].to_list()
        for bticker in bsymbols:
            st.write(f"⏳ {index,bticker} downloaded")
            index += 1
            df=yf.download(bticker,period="1y")
            df2=df.drop('Adj Close', 1)
            df3=df2.reset_index()
            df4=df3.round(2)
            df4.to_sql(bticker,engine, if_exists='replace')
            dfw=yf.download(bticker,period="250wk",interval = "1wk")
            df2w=dfw.drop('Adj Close', 1)
            df3w=df2w.reset_index()
            df4w=df3w.round(2)
            df4w.to_sql(bticker,enginew, if_exists='replace')
        now=pd.Timestamp.now().strftime("%d-%m-%Y, %H:%M")
        st.write('Last downloaded', index,ticker,now)
        return(index,ticker,now)
lastindex=getdata()
end = time.perf_counter() 
st.write('Last downloaded', lastindex, 'Süre', end - start)
def MACDdecision(df):
    df['MACD_diff']= ta.trend.macd_diff(df.Close)
    df['MACD']= ta.trend.macd(df.Close)
    df['MACD_signal']=ta.trend.macd_signal(df.Close)
    df.loc[(df.MACD_diff>0) & (df.MACD_diff.shift(1)<0),'Dec_MACD']='Buy'
    df.loc[(df.MACD_diff<0) & (df.MACD_diff.shift(1)>0),'Dec_MACD']='Sell'
    df.loc[(df.MACD_diff.shift(1)<df.MACD_diff),'Trend MACD']='Buy'
    df.loc[(df.MACD_diff.shift(1)>df.MACD_diff),'Trend MACD']='Sell'

def EMA_decision(df):

    df['EMA20'] = ta.trend.ema_indicator(df.Close,window=20)
    df.loc[(df.Close>df['EMA20']), 'Dec_EMA20'] = 'Buy'
    df.loc[(df.Close<df['EMA20']), 'Dec_EMA20'] = 'Sell'
    df.loc[((df.Close>=df.EMA20)& (df.Close.shift(1)<=df.EMA20.shift(1))), 'EMA20_cross'] = 'Buy'
    df.loc[(df.Close.shift(1)>=df.EMA20.shift(1))&(df.Low<=df.EMA20)&(df.Close>=df.EMA20), 'EMA20_cross'] = 'Buy2'
    df.loc[(df.Close.shift(1)<=df.EMA20.shift(1))&(df.High>=df.EMA20)&(df.Close<=df.EMA20), 'EMA20_cross'] = 'Sell2' 
    df.loc[((df.Close<=df.EMA20)& (df.Close.shift(1)>=df.EMA20.shift(1))), 'EMA20_cross'] = 'Sell'


    df['EMA50'] = ta.trend.ema_indicator(df.Close,window=50)
    df.loc[(df.Close>df['EMA50']), 'Dec_EMA50'] = 'Buy'
    df.loc[(df.Close<df['EMA50']), 'Dec_EMA50'] = 'Sell'
    df.loc[((df.Close>=df.EMA50)& (df.Close.shift(1)<=df.EMA50.shift(1))), 'EMA50_cross'] = 'Buy'
    df.loc[(df.Close.shift(1)>=df.EMA50.shift(1))&(df.Low<=df.EMA50)&(df.Close>=df.EMA50), 'EMA50_cross'] = 'Buy2'
    df.loc[(df.Close.shift(1)<=df.EMA50.shift(1))&(df.High>=df.EMA50)&(df.Close<=df.EMA50), 'EMA50_cross'] = 'Sell2' 
    df.loc[((df.Close<=df.EMA50)& (df.Close.shift(1)>=df.EMA50.shift(1))), 'EMA50_cross'] = 'Sell'


    df['EMA200'] = ta.trend.ema_indicator(df.Close,window=200)
    df.loc[(df.Close>df['EMA200']), 'Dec_EMA200'] = 'Buy'
    df.loc[(df.Close<df['EMA200']), 'Dec_EMA200'] = 'Sell'
    df.loc[((df.Close>=df.EMA200)& (df.Close.shift(1)<=df.EMA200.shift(1))), 'EMA200_cross'] = 'Buy'
    df.loc[((df.Close<=df.EMA200)& (df.Close.shift(1)>=df.EMA200.shift(1))), 'EMA200_cross'] = 'Sell'

def ADX_decision(df):
    df['ADX']= ta.trend.adx(df.High, df.Low, df.Close)
    df['ADX_neg']=ta.trend.adx_neg(df.High, df.Low, df.Close)
    df['ADX_pos']=ta.trend.adx_pos(df.High, df.Low, df.Close)
    df['DIOSQ']=df['ADX_pos']-df['ADX_neg']
    df['DIOSQ_EMA']=ta.trend.ema_indicator(df.DIOSQ,window=10)
    df.loc[(df.ADX>df.ADX.shift(1)) ,'Decision ADX']='Buy'
    df.loc[(df.DIOSQ>df.DIOSQ_EMA)& (df.DIOSQ.shift(1)<df.DIOSQ_EMA.shift(1)), 'Dec_DIOSQ'] = 'Buy'
    df.loc[(df.DIOSQ<df.DIOSQ_EMA)& (df.DIOSQ.shift(1)>df.DIOSQ_EMA.shift(1)), 'Dec_DIOSQ'] = 'Sell'

def Supertrend(df):
    df['sup']=pa.supertrend(high=df['High'],low=df['Low'],close=df['Close'],length=10,multiplier=1.0)['SUPERTd_10_1.0']
    df['sup2']=pa.supertrend(high=df['High'],low=df['Low'],close=df['Close'],length=10,multiplier=1.0)['SUPERT_10_1.0']
    df['sup3']=pa.supertrend(high=df['High'],low=df['Low'],close=df['Close'],length=10,multiplier=2.0)['SUPERTd_10_2.0']
    df['sup4']=pa.supertrend(high=df['High'],low=df['Low'],close=df['Close'],length=10,multiplier=2.0)['SUPERT_10_2.0']
    df['sup5']=pa.supertrend(high=df['High'],low=df['Low'],close=df['Close'],length=10,multiplier=3.0)['SUPERTd_10_3.0']
    df['sup6']=pa.supertrend(high=df['High'],low=df['Low'],close=df['Close'],length=10,multiplier=3.0)['SUPERT_10_3.0']
    
    
    df.loc[(df.sup==1)&(df.sup.shift(1)==-1), 'Decision Super'] = 'Buy'
    df.loc[(df.Close.shift(1)>=df.sup2.shift(1))&(df.Low<=df.sup2)&(df.Close>df.sup2), 'Decision Super'] = 'Buy2'    
    df.loc[(df.Close.shift(1)<=df.sup2.shift(1))&(df.High>=df.sup2)&(df.Close<df.sup2), 'Decision Super'] = 'Sell2'
    df.loc[(df.sup==-1)&(df.sup.shift(1)==1), 'Decision Super'] = 'Sell' 

    
    
    df.loc[(df.sup3==1)&(df.sup3.shift(1)==-1), 'Decision Super2'] = 'Buy'
    df.loc[(df.Close.shift(1)>=df.sup4.shift(1))&(df.Low<=df.sup4)&(df.Close>df.sup4), 'Decision Super2'] = 'Buy2'    
    df.loc[(df.Close.shift(1)<=df.sup4.shift(1))&(df.High>=df.sup4)&(df.Close<df.sup4), 'Decision Super2'] = 'Sell2'
    df.loc[(df.sup3==-1)&(df.sup3.shift(1)==1), 'Decision Super2'] = 'Sell'

    
    df.loc[(df.sup5==1)&(df.sup5.shift(1)==-1), 'Decision Super3'] = 'Buy'
    df.loc[(df.Close.shift(1)>=df.sup6.shift(1))&(df.Low<=df.sup6)&(df.Close>df.sup6), 'Decision Super3'] = 'Buy2'    
    df.loc[(df.Close.shift(1)<=df.sup6.shift(1))&(df.High>=df.sup6)&(df.Close<df.sup6), 'Decision Super3'] = 'Sell2'
    df.loc[(df.sup5==-1)&(df.sup5.shift(1)==1), 'Decision Super3'] = 'Sell' 

    
    df.loc[(df.sup2 == df.sup2.shift(1)), 'Consolidating'] = 'Yes'
    df.loc[(df.sup4 == df.sup4.shift(1)), 'Consolidating2'] = 'Yes'
    df.loc[(df.sup6 == df.sup6.shift(1)), 'Consolidating3'] = 'Yes'
def ATR_decision(df):
    df['ATR']= ta.volatility.average_true_range(df.High, df.Low, df.Close,window=10)
    df['ATR%'] = df['ATR']/df.Close*100
    df['RISK']= 2*df['ATR']/701*100        

# def Stoch_decision(df):
#     df['Stoch'] = ta.momentum.stoch(df.High, df.Low, df.Close, smooth_window=3)
#     df['Stoch_Signal'] = ta.momentum.stoch_signal(df.High, df.Low, df.Close, smooth_window=3)
#     df.loc[(df.Stoch>df.Stoch_Signal)& (df.Stoch.shift(1)<df.Stoch_Signal.shift(1)) & (df.Stoch_Signal<20), 'Decision Stoch'] = 'Buy'  

def Stochrsi_decision(df):
     df['Stochrsi_d'] = ta.momentum.stochrsi_d(df.Close)
     df['Stochrsi_k'] = ta.momentum.stochrsi_k(df.Close)
     #df.loc[(df.Stochrsi_k.shift(1)>0.8)&(df.Stochrsi_k<0.8),'DecStoch']='Sell'

def Volume_decision(df):
    df['Volume_EMA']=ta.trend.ema_indicator(df.Volume,window=10)


@st.cache(allow_output_mutation=True)
def connect_engine(url):
    engine=sqlalchemy.create_engine(url) 
    return engine
@st.cache(allow_output_mutation=True)
def connect_enginew(url):
    enginew=sqlalchemy.create_engine(url) 
    return enginew
start = time.perf_counter()
def get_names():
    names= pd.read_sql('SELECT name FROM sqlite_master WHERE type="table"',engine)
    names = names.name.to_list()
    return names
    
@st.cache(hash_funcs={sqlalchemy.engine.base.Engine:id},suppress_st_warning=True,max_entries=2)
def get_framelist():
    framelist=[]
    for name in names:
        framelist.append(pd.read_sql(f'SELECT Date,Close,Open,High,Low,Volume FROM "{name}"',engine))    
    np.seterr(divide='ignore', invalid='ignore')
    with st.empty():
        sira=0
        for name,frame in zip(names,framelist): 
            if len(frame)>30:
                MACDdecision(frame)
                EMA_decision(frame)
                ADX_decision(frame)
                Supertrend(frame)
                ATR_decision(frame)
                Stochrsi_decision(frame)
                Volume_decision(frame)
                sira +=1
                st.write('günlük',sira,name)             
    return framelist    
@st.cache(hash_funcs={sqlalchemy.engine.base.Engine:id},suppress_st_warning=True,max_entries=2)      
def get_framelistw():
    framelistw=[]
    for name in names: 
        framelistw.append(pd.read_sql(f'SELECT Date,Close,Open,High,Low,Volume FROM "{name}"',enginew))   
    np.seterr(divide='ignore', invalid='ignore')
    with st.empty():
        sira=0
        for name,framew in zip(names,framelistw): 
            if  len(framew)>30 :
                MACDdecision(framew)
                EMA_decision(framew)
                ADX_decision(framew)
                Supertrend(framew)
                ATR_decision(framew)
                Stochrsi_decision(framew)
                Volume_decision(framew)
                sira +=1
                st.write('haftalik',sira,name)              
    return framelistw        
connection_url='sqlite:///günlük.db'
connection_url2='sqlite:///haftalik.db'
engine= connect_engine(connection_url) 
enginew= connect_enginew(connection_url2) 
start = time.perf_counter()
names=get_names()
framelist=get_framelist() 
framelistw=get_framelistw()
end = time.perf_counter()
st.write(end - start)

def get_figures(frame,r):
    fig = go.Figure()
    fig = plotly.subplots.make_subplots(rows=3, cols=1, shared_xaxes=True,
    vertical_spacing=0.01, row_heights=[0.5,0.2,0.2])
    fig.add_trace(go.Candlestick(x=frame['Date'].tail(r), open=frame['Open'].tail(r), high=frame['High'].tail(r), low=frame['Low'].tail(r), close=frame['Close'].tail(r)))
    fig.add_trace(go.Scatter(x=frame['Date'].tail(r), 
         y=frame['EMA20'].tail(r), 
         opacity=0.7, 
         line=dict(color='green', width=2), 
         name='EMA 20'))
    fig.add_trace(go.Scatter(x=frame['Date'].tail(r), 
         y=frame['EMA50'].tail(r), 
         opacity=0.7, 
         line=dict(color='orange', width=2), 
         name='EMA 50'))
    fig.add_trace(go.Scatter(x=frame['Date'].tail(r), 
         y=frame['EMA200'].tail(r), 
         opacity=0.7, 
         line=dict(color='blue', width=2), 
         name='EMA 200'))
    fig.add_trace(go.Scatter(x=frame['Date'].tail(r), 
         y=frame['sup2'].tail(r),
         opacity=0.7,
         mode='markers', marker=dict(size=3,color='green'), 
         name='Supertrend1'))
    fig.add_trace(go.Scatter(x=frame['Date'].tail(r), 
         y=frame['sup4'].tail(r),
         opacity=0.7,
         mode='markers', marker=dict(size=3,color='orange'), 
         name='Supertrend2'))
    fig.add_trace(go.Scatter(x=frame['Date'].tail(r), 
         y=frame['sup6'].tail(r),
         opacity=0.7,
         mode='markers', marker=dict(size=3,color='blue'), 
         name='Supertrend3'))
    fig.add_trace(go.Bar(x=frame['Date'].tail(r), 
     y=frame['MACD_diff'].tail(r)
        ), row=2, col=1)
    fig.add_trace(go.Scatter(x=frame['Date'].tail(r),
         y=frame['MACD'].tail(r),
         line=dict(color='blue', width=1)
        ), row=2, col=1)
    fig.add_trace(go.Scatter(x=frame['Date'].tail(r),
         y=frame['MACD_signal'].tail(r),
         line=dict(color='orange', width=1)
        ), row=2, col=1)
    fig.add_trace(go.Bar(x=frame['Date'].tail(r), 
     y=frame['Volume'].tail(r)
        ), row=3, col=1)
    fig.add_trace(go.Scatter(x=frame['Date'].tail(r),
         y=frame['Volume_EMA'].tail(r),
         line=dict(color='orange', width=2)
        ), row=3, col=1)
    fig.add_hline(y=0.2, line_width=1, line_dash="dash", line_color="green",row=3, col=1)
    fig.add_hline(y=0.5, line_width=1, line_dash="dash", line_color="green",row=3, col=1)
    fig.add_hline(y=0.8, line_width=1, line_dash="dash", line_color="green",row=3, col=1)
    fig.update_layout( height=600, width=1200,
        showlegend=False, xaxis_rangeslider_visible=False)
    return fig
def expander():
    with st.expander(str(sira) +') '+ name+'/'+' RISK= '+str(frame['RISK'].iloc[-1].round(2))+'/ %ATR='+str(frame['ATR%'].iloc[-1].round(2))):
        #st.write(str(sira) +') '+ name+'/'+' RISK= '+str(frame['RISK'].iloc[-1].round(2))+'/ %ATR='+str(frame['ATR%'].iloc[-1].round(2)))
        col3, col4 = st.columns([1, 1])
        col3.write(frame[['Close','EMA20_cross','EMA50_cross','Decision Super','Decision Super2','Decision Super3','Dec_MACD','ADX','Trend MACD','MACD_diff']].tail(2))
        col4.write(framew[['Close','ATR%','ADX','Dec_EMA50','Dec_MACD','Trend MACD','MACD_diff']].tail(2))
        col1, col2 = st.columns([1, 1])
        r=200
        fig=get_figures(frame,r)
        r=40
        figw=get_figures(framew,r)
        col1.plotly_chart(fig,use_container_width=True)
        col2.plotly_chart(figw,use_container_width=True)        
sira=0
option1 = st.sidebar.selectbox("Buy or Sell",('Buy','Sell')) 
option2 = st.sidebar.selectbox("Which Indicator?", ('EMASUPER','Index','EMA50','Supertrend','EMA20','MACD','ADX','Consolidating','EMA200'))
adx_value= st.sidebar.number_input('ADX Value',min_value=10,value=18)
adx_value2= st.sidebar.number_input('ADX Value_ust',min_value=10,value=25)
riskvalue=st.sidebar.number_input('Risk',min_value=1,value=1000)
option3=st.sidebar.text_input('Ticker','Enter Ticker Name')
fark=st.sidebar.number_input('Fark',min_value=1.0,value=5.0,step=0.5)
st.header(option1 + option2)
indices=['US500/USD_S&P 500_INDEX_US','EU50/EUR_Euro Stoxx 50_INDEX_DE','^N225','XU030.IS']
for name, frame,framew in zip(names,framelist,framelistw): 
    try:
        if  len(frame)>30 and len(framew)>30 and frame['ADX'].iloc[-1]>=adx_value and frame['ADX'].iloc[-1]<=adx_value2 and frame['RISK'].iloc[-1]<=riskvalue:
            
            if option1 == 'Buy' and (framew['Dec_EMA20'].iloc[-1]=='Buy' or framew['Dec_EMA50'].iloc[-1]=='Buy'\
            or framew['Close'].iloc[-1]>framew['sup4'].iloc[-1] or framew['Close'].iloc[-1]>framew['sup6'].iloc[-1]\
            or framew['Close'].iloc[-1]>framew['sup2'].iloc[-1]):
            #or framew['Trend MACD'].iloc[-1]=='Buy'
           
                if option2 == 'EMASUPER':
                    if (frame['Decision Super2'].iloc[-1]=='Buy' or frame['EMA50_cross'].iloc[-1]=='Buy'\
                    or (frame['Decision Super'].iloc[-1]=='Buy' and frame['Consolidating'].iloc[-2]=='Yes'))\
                    and frame['EMA20'].iloc[-1]<frame['EMA50'].iloc[-1]:
                    #and (frame['Dec_EMA50'].iloc[-1]=='Buy' or frame['Dec_EMA20'].iloc[-1]=='Buy'):
                    #and (frame['Close'].iloc[-1]>frame['sup4'].iloc[-1] or frame['Close'].iloc[-1]>frame['sup6'].iloc[-1])
                    #or frame['Decision Super'].iloc[-1]=='Buy2' or frame['Decision Super2'].iloc[-1]=='Buy2' or frame['Decision Super3'].iloc[-1]=='Buy2'\
                    #or frame['EMA50_cross'].iloc[-1]=='Buy2' or frame['EMA20_cross'].iloc[-1]=='Buy2')\
                    #or frame['Decision Super3'].iloc[-1]=='Buy' or frame['EMA20_cross'].iloc[-1]=='Buy' or frame['EMA50_cross'].iloc[-1]=='Buy')\
                            sira +=1
                            expander()
                    #elif ((frame['Decision Super'].iloc[-1]=='Buy2' or frame['Decision Super2'].iloc[-1]=='Buy2' or frame['Decision Super3'].iloc[-1]=='Buy2'\
                    #or frame['EMA50_cross'].iloc[-1]=='Buy2' or frame['EMA20_cross'].iloc[-1]=='Buy2') and frame['EMA20'].iloc[-1]>frame['EMA50'].iloc[-1]\
                    #and frame['Dec_EMA50'].iloc[-1]=='Buy'):
                    #        sira +=1
                    #        expander()

                if option2 == 'Consolidating':
                    if (frame['Consolidating'].iloc[-1]=='Yes' and frame['Consolidating2'].iloc[-1]=='Yes' and frame['Consolidating3'].iloc[-1]=='Yes')\
                    and (frame['EMA20'].iloc[-1]>frame['EMA50'].iloc[-1] or frame['Dec_EMA50'].iloc[-1]=='Buy')\
                    and (frame['Close'].iloc[-1]>frame['sup4'].iloc[-1] or frame['Close'].iloc[-1]>frame['sup6'].iloc[-1]):
                    #and (frame['Dec_EMA20'].iloc[-1]=='Buy' and frame['Dec_EMA50'].iloc[-1]=='Sell'):
                    #and (frame['Close'].iloc[-1]<frame['sup6'].iloc[-1] or frame['Close'].iloc[-1]<frame['sup4'].iloc[-1] or frame['Close'].iloc[-1]<frame['sup4'].iloc[-1])
                            sira +=1
                            expander()   
            if option1 == 'Sell' and (framew['Dec_EMA20'].iloc[-1]=='Sell' or framew['Dec_EMA50'].iloc[-1]=='Sell'\
            or framew['Close'].iloc[-1]<framew['sup4'].iloc[-1] or framew['Close'].iloc[-1]<framew['sup6'].iloc[-1]\
            or framew['Close'].iloc[-1]<framew['sup2'].iloc[-1]):
            #and (framew['Dec_EMA20'].iloc[-1]=='Sell' or framew['Dec_EMA50'].iloc[-1]=='Sell')
            #and (framew['Close'].iloc[-1]<framew['sup4'].iloc[-1] or framew['Close'].iloc[-1]<framew['sup6'].iloc[-1])
                if option2 == 'EMASUPER':
                   if (frame['Decision Super2'].iloc[-1]=='Sell'\
                   or (frame['Decision Super'].iloc[-1]=='Sell' and frame['Consolidating'].iloc[-2]=='Yes'))\
                   and frame['EMA20'].iloc[-1]>frame['EMA50'].iloc[-1]\
                   and (frame['Dec_EMA50'].iloc[-1]=='Sell' or frame['Dec_EMA20'].iloc[-1]=='Sell'):
                   #and (frame['Close'].iloc[-1]<frame['sup4'].iloc[-1] or frame['Close'].iloc[-1]<frame['sup6'].iloc[-1]):
                   #and frame['EMA50'].iloc[-1]<(1+(fark/100))*frame['EMA20'].iloc[-1]
                   #and (frame['Close'].iloc[-1]<frame['sup2'].iloc[-1] or frame['Close'].iloc[-1]<frame['sup4'].iloc[-1]\
                            sira +=1
                            expander()
                   #elif ((frame['Decision Super2'].iloc[-1]=='Sell2' or frame['Decision Super3'].iloc[-1]=='Sell2'\
                   #or frame['EMA50_cross'].iloc[-1]=='Sell2' or frame['EMA20_cross'].iloc[-1]=='Sell2') and frame['EMA20'].iloc[-1]<frame['EMA50'].iloc[-1]\
                   #and (frame['Close'].iloc[-1]<frame['sup6'].iloc[-1] or frame['Close'].iloc[-1]<frame['sup4'].iloc[-1])):
                   #         sira +=1
                   #         expander() 
                if option2 == 'Consolidating':
                    if (frame['Consolidating2'].iloc[-1]=='Yes' and frame['Consolidating3'].iloc[-1]=='Yes')\
                    and (frame['EMA20'].iloc[-1]<frame['EMA50'].iloc[-1] and frame['Dec_EMA20'].iloc[-1]=='Buy')\
                    and frame['Close'].iloc[-1]<frame['sup4'].iloc[-1]<frame['sup6'].iloc[-1]:
                    #and (frame['Dec_EMA20'].iloc[-1]=='Buy' and frame['Dec_EMA50'].iloc[-1]=='Sell'):
                    #and (frame['Close'].iloc[-1]<frame['sup6'].iloc[-1] or frame['Close'].iloc[-1]<frame['sup4'].iloc[-1] or frame['Close'].iloc[-1]<frame['sup4'].iloc[-1])
                            sira +=1
                            expander()   
        if option2 == 'Index' and name in indices:
                sira +=1
                expander()
        if name in option3:
                sira +=1
                expander()
    except Exception as e:
        st.write(name,e) 
        
    
