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
    df.loc[((df.Close>=df.EMA20)& (df.Close.shift(1)<=df.EMA20.shift(1)))|((df.Close.shift(1)>=df.EMA20.shift(1))& \
    (df.Low<=df.EMA20)), 'EMA20_cross'] = 'Buy'
    df.loc[((df.Close<=df.EMA20)& (df.Close.shift(1)>=df.EMA20.shift(1)))|((df.Close.shift(1)<=df.EMA20.shift(1))& \
    (df.High>=df.EMA20)), 'EMA20_cross'] = 'Sell'

    df['EMA50'] = ta.trend.ema_indicator(df.Close,window=50)
    df.loc[(df.Close>df['EMA50']), 'Dec_EMA50'] = 'Buy'
    df.loc[(df.Close<df['EMA50']), 'Dec_EMA50'] = 'Sell'
    #df.loc[((df.Close>df.EMA50)& (df.Close.shift(1)>df.EMA50.shift(1))), 'Dec_EMA50'] = 'Buy'
    #df.loc[((df.Close<df.EMA50)& (df.Close.shift(1)<df.EMA50.shift(1))), 'Dec_EMA50'] = 'Sell'
    df.loc[((df.Close>=df.EMA50)& (df.Close.shift(1)<=df.EMA50.shift(1)))|((df.Close.shift(1)>=df.EMA50.shift(1))& \
    (df.Low<=df.EMA50)), 'EMA50_cross'] = 'Buy'
    df.loc[((df.Close<=df.EMA50)& (df.Close.shift(1)>=df.EMA50.shift(1)))|((df.Close.shift(1)<=df.EMA50.shift(1))& \
    (df.High>=df.EMA50)), 'EMA50_cross'] = 'Sell'


    df['EMA200'] = ta.trend.ema_indicator(df.Close,window=200)
    df.loc[(df.Close>df['EMA200']), 'Dec_EMA200'] = 'Buy'
    df.loc[(df.Close<df['EMA200']), 'Dec_EMA200'] = 'Sell'
    df.loc[((df.Close>=df.EMA200)& (df.Close.shift(1)<=df.EMA200.shift(1)))|((df.Close.shift(1)>=df.EMA200.shift(1))& \
    (df.Low<=df.EMA200)), 'EMA200_cross'] = 'Buy'
    df.loc[((df.Close<=df.EMA200)& (df.Close.shift(1)>=df.EMA200.shift(1)))|((df.Close.shift(1)<=df.EMA200.shift(1))& \
    (df.High>=df.EMA200)), 'EMA200_cross'] = 'Sell'

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
    df.loc[(df.sup3==1)&(df.sup3.shift(1)==-1), 'Decision Super2'] = 'Buy'
    df.loc[(df.sup3==-1)&(df.sup3.shift(1)==1), 'Decision Super2'] = 'Sell'  
    df.loc[(df.sup==1)&(df.sup.shift(1)==-1)|((df.Close.shift(1)>=df.sup2.shift(1))& \
    (df.Low<=df.sup2)&(df.Close>df.sup2)), 'Decision Super'] = 'Buy'
    df.loc[(df.sup==-1)&(df.sup.shift(1)==1), 'Decision Super'] = 'Sell' 
    df.loc[(df.sup5==1)&(df.sup5.shift(1)==-1), 'Decision Super3'] = 'Buy'
    df.loc[(df.sup5==-1)&(df.sup5.shift(1)==1), 'Decision Super3'] = 'Sell' 
    df.loc[(df.sup2 == df.sup2.shift(8))&(df.sup2 != df.sup2.shift(9)), 'Consolidating'] = 'Yes'
    df.loc[(df.sup4 == df.sup4.shift(8))&(df.sup4 != df.sup4.shift(9)), 'Consolidating2'] = 'Yes'
    df.loc[(df.sup6 == df.sup6.shift(3)), 'Consolidating3'] = 'Yes'
def ATR_decision(df):
    df['ATR']= ta.volatility.average_true_range(df.High, df.Low, df.Close,window=10)
    df['ATR%'] = df['ATR']/df.Close*100
    df['RISK']= 2*df['ATR']/700*100        

# def Stoch_decision(df):
#     df['Stoch'] = ta.momentum.stoch(df.High, df.Low, df.Close, smooth_window=3)
#     df['Stoch_Signal'] = ta.momentum.stoch_signal(df.High, df.Low, df.Close, smooth_window=3)
#     df.loc[(df.Stoch>df.Stoch_Signal)& (df.Stoch.shift(1)<df.Stoch_Signal.shift(1)) & (df.Stoch_Signal<20), 'Decision Stoch'] = 'Buy'  



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
        framelist.append(pd.read_sql(f'SELECT Date,Close,Open,High,Low FROM "{name}"',engine))    
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
                sira +=1
                st.write('günlük',sira,name)             
    return framelist    
@st.cache(hash_funcs={sqlalchemy.engine.base.Engine:id},suppress_st_warning=True,max_entries=2)      
def get_framelistw():
    framelistw=[]
    for name in names: 
        framelistw.append(pd.read_sql(f'SELECT Date,Close,Open,High,Low FROM "{name}"',enginew))   
    np.seterr(divide='ignore', invalid='ignore')
    with st.empty():
        sira=0
        for name,framew in zip(names,framelistw): 
            if  len(framew)>30 :
                MACDdecision(framew)
                EMA_decision(framew)
                ADX_decision(framew)
                Supertrend(framew)
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

def get_figures(frame):
    fig = go.Figure()
    fig = plotly.subplots.make_subplots(rows=3, cols=1, shared_xaxes=True,
    vertical_spacing=0.01, row_heights=[0.5,0.2,0.2])
    r=50
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
         mode='markers', marker=dict(size=2,color='green'), 
         name='Supertrend1'))
    fig.add_trace(go.Scatter(x=frame['Date'].tail(r), 
         y=frame['sup4'].tail(r),
         opacity=0.7,
         mode='markers', marker=dict(size=2,color='orange'), 
         name='Supertrend2'))
    fig.add_trace(go.Scatter(x=frame['Date'].tail(r), 
         y=frame['sup6'].tail(r),
         opacity=0.7,
         mode='markers', marker=dict(size=2,color='blue'), 
         name='Supertrend3'))
    fig.add_trace(go.Bar(x=frame['Date'].tail(r), 
     y=frame['MACD_diff'].tail(r)
        ), row=2, col=1)
    fig.add_trace(go.Scatter(x=frame['Date'].tail(r),
         y=frame['MACD'].tail(r),
         line=dict(color='black', width=2)
        ), row=2, col=1)
    fig.add_trace(go.Scatter(x=frame['Date'].tail(r),
         y=frame['MACD_signal'].tail(r),
         line=dict(color='blue', width=1)
        ), row=2, col=1)
    fig.add_trace(go.Scatter(x=frame['Date'].tail(r),
         y=frame['ADX'].tail(r),
         line=dict(color='orange', width=1)
        ), row=3, col=1)
    fig.add_trace(go.Scatter(x=frame['Date'].tail(r),
         y=frame['DIOSQ'].tail(r),
         line=dict(color='green', width=1)
        ), row=3, col=1)
    fig.add_trace(go.Scatter(x=frame['Date'].tail(r),
         y=frame['DIOSQ_EMA'].tail(r),
         line=dict(color='purple', width=1)
        ), row=3, col=1)   
    fig.update_layout( height=600, width=1200,
        showlegend=False, xaxis_rangeslider_visible=False)
    return fig
def expander():
    with st.expander(str(sira) +') '+ name+'/'+' RISK= '+str(frame['RISK'].iloc[-1].round(2))+'/ %ATR='+str(frame['ATR%'].iloc[-1].round(2))):
        col3, col4 = st.columns([1, 1])
        col3.write(frame[['Close','RISK','ATR%','ADX','EMA20_cross','EMA50_cross','EMA200_cross','Dec_MACD','Dec_DIOSQ','Trend MACD','MACD_diff']].tail(2))
        col4.write(framew[['Close','sup2','Dec_EMA50','Dec_MACD','Trend MACD','MACD_diff']].tail(2))
        col1, col2 = st.columns([1, 1])
        fig=get_figures(frame)
        figw=get_figures(framew)
        col1.plotly_chart(fig,use_container_width=True)
        col2.plotly_chart(figw,use_container_width=True)
sira=0
option1 = st.sidebar.selectbox("Buy or Sell",('Buy','Sell')) 
option2 = st.sidebar.selectbox("Which Indicator?", ('EMA50','Supertrend','EMA20','MACD','ADX','Consolidating','Index','EMA200'))
adx_value= st.sidebar.number_input('ADX Value',min_value=10,value=18)
adx_value2= st.sidebar.number_input('ADX Value_ust',min_value=10,value=25)
riskvalue=st.sidebar.number_input('Risk',min_value=0.01,value=1.0,step=0.1)
st.header(option1 + option2)
indices=['US500/USD_S&P 500_INDEX_US','EU50/EUR_Euro Stoxx 50_INDEX_DE','^N225']
for name, frame,framew in zip(names,framelist,framelistw): 
    try:
        if len(frame)>30 and len(framew)>30 and frame['ADX'].iloc[-1]>=adx_value and frame['RISK'].iloc[-1]<=riskvalue :
            if option1 == 'Buy' and framew['Trend MACD'].iloc[-1]=='Buy' and (framew['Dec_EMA50'].iloc[-1]=='Buy' or framew['MACD_diff'].iloc[-1]>0) : 
            #and framew['sup'].iloc[-1]==1 and framew['Dec_EMA50'].iloc[-1]=='Buy':
                if option2 == 'EMA50':  
                    if frame['EMA50_cross'].iloc[-1]=='Buy' and frame['MACD_diff'].iloc[-1]>0:
                            sira +=1
                            expander()
                if option2 == 'EMA200':  
                    if frame['EMA200_cross'].iloc[-1]=='Buy' and frame['MACD_diff'].iloc[-1]>0:
                            sira +=1
                            expander()
                if option2 == 'EMA20':
                    if frame['EMA20_cross'].iloc[-1]=='Buy' and frame['MACD_diff'].iloc[-1]>0:
                            sira +=1
                            expander() 
                if option2 == 'ADX':
                    if frame['Decision ADX'].iloc[-1]=='Buy' and frame['ADX'].iloc[-1]<=adx_value2 and frame['MACD_diff'].iloc[-1]>0:
                            sira +=1
                            expander()
                if option2 == 'MACD':
                    if frame['Dec_MACD'].iloc[-1]=='Buy' and frame['MACD'].iloc[-1]<=0 :
                            sira +=1
                            expander()
                if option2 == 'Consolidating':
                    if frame['Consolidating3'].iloc[-1]=='Yes' and frame['MACD_diff'].iloc[-1]>0:
                    #and frame['MACD_diff'].iloc[-1]>0 and framew['Trend MACD'].iloc[-1]=='Buy' and frame['Dec_EMA50'].iloc[-1]=='Sell':
                            sira +=1
                            expander()          
                if option2 == 'Supertrend':
                    if frame['Decision Super'].iloc[-1]=='Buy' or frame['Decision Super2'].iloc[-1]=='Buy' or frame['Decision Super3'].iloc[-1]=='Buy'\
                    and frame['MACD_diff'].iloc[-1]>0:
                            sira +=1
                            expander()
               # if option2 == 'Supertrend2':
               #     if frame['Decision Super2'].iloc[-1]=='Buy' :
               #             sira +=1
               #             expander()
               # if option2 == 'Supertrend3':
               #     if frame['Decision Super3'].iloc[-1]=='Buy' :
               #             sira +=1
               #             expander()
            elif option1 == 'Sell' and framew['Trend MACD'].iloc[-1]=='Sell' and (framew['Dec_EMA50'].iloc[-1]=='Sell' or framew['MACD_diff'].iloc[-1]<0):
            #and framew['sup'].iloc[-1]==-1 and framew['Dec_EMA50'].iloc[-1]=='Sell':
                if option2 == 'EMA50':  
                    if frame['EMA50_cross'].iloc[-1]=='Sell' and frame['MACD_diff'].iloc[-1]<0 :
                            sira +=1
                            expander()
                if option2 == 'EMA200':  
                    if frame['EMA200_cross'].iloc[-1]=='Sell' and frame['MACD_diff'].iloc[-1]<0:
                            sira +=1
                            expander()
                if option2 == 'EMA20':
                    if frame['EMA20_cross'].iloc[-1]=='Sell' and frame['MACD_diff'].iloc[-1]<0 :
                            sira +=1
                            expander() 
                if option2 == 'ADX':
                    if frame['Decision ADX'].iloc[-1]=='Sell' and frame['ADX'].iloc[-1]<=adx_value2 and frame['MACD_diff'].iloc[-1]<0:
                            sira +=1
                            expander()
                if option2 == 'MACD':
                    if frame['Dec_MACD'].iloc[-1]=='Sell' and frame['MACD'].iloc[-1]>=0:
                            sira +=1
                            expander()
                if option2 == 'Consolidating':
                    if frame['Consolidating3'].iloc[-1]=='Yes' and frame['MACD_diff'].iloc[-1]<0 :
                    #and frame['MACD_diff'].iloc[-1]<0 and framew['Trend MACD'].iloc[-1]=='Sell' and frame['Dec_EMA50'].iloc[-1]=='Buy':
                            sira +=1
                            expander()          
                if option2 == 'Supertrend':
                    if frame['Decision Super'].iloc[-1]=='Sell' or frame['Decision Super2'].iloc[-1]=='Sell' or frame['Decision Super3'].iloc[-1]=='Sell'\
                    and frame['MACD_diff'].iloc[-1]<0:
                            sira +=1
                            expander()   
                #if option2 == 'Supertrend2':
                #    if frame['Decision Super2'].iloc[-1]=='Sell' :
                #            sira +=1
                #            expander()
                #if option2 == 'Supertrend3':
                #    if frame['Decision Super3'].iloc[-1]=='Sell' :
                #            sira +=1
                #            expander() 
        if option2 == 'Index' and name in indices:
                sira +=1
                expander()            
    except Exception as e:
        st.write(name,e) 
    
   
