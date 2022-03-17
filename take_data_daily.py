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
                data2 = exchange.fetch_ohlcv(ticker, timeframe='1d',limit=155) #since=exchange.parse8601('2022-02-13T00:00:00Z'))
                data3= exchange.fetch_ohlcv(ticker, timeframe='1w',limit=55)
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
            #print(index,bticker,end="\r")
            st.write(f"⏳ {index,bticker} downloaded")
            index += 1
            df=yf.download(bticker,period="1y")
            df2=df.drop('Adj Close', 1)
            df3=df2.reset_index()
            df4=df3.round(2)
            df4.to_sql(bticker,engine, if_exists='replace')
            dfw=yf.download(bticker,period="55wk",interval = "1wk")
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
    df.loc[(df.MACD_diff>0) & (df.MACD_diff.shift(1)<0),'Decision MACD']='Buy'
    df.loc[(df.MACD_diff<0) & (df.MACD_diff.shift(1)>0),'Decision MACD']='Sell'
    df.loc[(df.MACD_diff.shift(1)<df.MACD_diff),'Trend MACD']='Strong'
def EMA_decision(df):
    df['EMA50'] = ta.trend.ema_indicator(df.Close,window=50)
    df.loc[(df.Close>df['EMA50']), 'Decision EMA50'] = 'Buy'
    df.loc[(df.Close<df['EMA50']), 'Decision EMA50'] = 'Sell'
    df.loc[(df.Close>df.EMA50)& (df.Close.shift(1)<df.EMA50.shift(1)), 'Decision EMA50_cross'] = 'Buy'
    df.loc[(df.Close<df.EMA50)& (df.Close.shift(1)>df.EMA50.shift(1)), 'Decision EMA50_cross'] = 'Sell'
    # df['EMA200'] = ta.trend.ema_indicator(df.Close,window=200)
    # df.loc[(df.Close>df['EMA200']), 'Decision EMA200'] = 'Buy'
    # df.loc[(df.Close<df['EMA200']), 'Decision EMA200'] = 'Sell'
    # df.loc[(df.Close>df.EMA200)& (df.Close.shift(1)<df.EMA200.shift(1)), 'Decision EMA200_cross'] = 'Buy'
    # df.loc[(df.Close<df.EMA200)& (df.Close.shift(1)>df.EMA200.shift(1)), 'Decision EMA200_cross'] = 'Sell'

def ADX_decision(df):
    df['ADX']= ta.trend.adx(df.High, df.Low, df.Close)
    #df.loc[(df.ADX>df.ADX.shift(1)) & (df.ADX>=18),'Decision ADX']='Buy'

def Supertrend(df):
    df['sup']=pa.supertrend(high=df['High'],low=df['Low'],close=df['Close'],length=10,multiplier=1)['SUPERTd_10_1.0']
    df.loc[(df.sup==1)&(df.sup.shift(1)==-1), 'Decision Super'] = 'Buy'
    df.loc[(df.sup==-1)&(df.sup.shift(1)==1), 'Decision Super'] = 'Sell'  


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
    
@st.cache(hash_funcs={sqlalchemy.engine.base.Engine:id},suppress_st_warning=True)
def get_framelist():
    framelist=[]
    for name in names:
        framelist.append(pd.read_sql(f'SELECT Date,Close,High,Low FROM "{name}"',engine))    
    np.seterr(divide='ignore', invalid='ignore')
    with st.empty():
        sira=0
        for name,frame in zip(names,framelist): 
            if len(frame)>30:
                MACDdecision(frame)
                EMA_decision(frame)
                ADX_decision(frame)
                #Supertrend(frame)
                # print(name)
                # print(frame)
                # print(name)
                sira +=1
                #print(sira)
                st.write('günlük',sira,name)             
    return framelist    
@st.cache(hash_funcs={sqlalchemy.engine.base.Engine:id},suppress_st_warning=True)      
def get_framelistw():
    framelistw=[]
    for name in names: 
        framelistw.append(pd.read_sql(f'SELECT Date,Close,High,Low FROM "{name}"',enginew))   
    np.seterr(divide='ignore', invalid='ignore')
    with st.empty():
        sira=0
        for name,framew in zip(names,framelistw): 
            if  len(framew)>30 :
                MACDdecision(framew)
                EMA_decision(framew)
                ADX_decision(framew)
                Supertrend(framew)
                #print(name)
                #print(framew)
                sira +=1
                #print(sira)
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

option1 = st.sidebar.selectbox("Buy or Sell",('Buy','Sell')) 
option2 = st.sidebar.selectbox("Which Indicator?", ('EMA', 'MACD'))#,'SUPERTREND'))
#option3= st.sidebar.selectbox("Day or Week", ('Day','Week'))
adx_value= st.sidebar.number_input('ADX Value',min_value=10,value=15)
st.header(option1+ option2)
sira=0
for name, frame,framew in zip(names,framelist,framelistw): 
    #if option3=='Day':
        if option1 == 'Buy'and option2 == 'EMA':  
            try:
                if len(frame)>30 and len(framew)>30 and framew['Decision EMA50'].iloc[-1]=='Buy' \
                and frame['ADX'].iloc[-1]>=adx_value and (frame['MACD_diff'].iloc[-1]>0 or frame['Trend MACD'].iloc[-1]=='Strong')  \
                and (framew['MACD_diff'].iloc[-1]>0 or framew['Trend MACD'].iloc[-1]=='Strong') and frame['Decision EMA50_cross'].iloc[-1]=='Buy' \ 
                and framew['sup'].iloc[-1]=='1': 
                    sira +=1
                    st.write(str(sira)+" Buying EMA50 for "+ name)
                    st.write(frame.tail(2))
                    #st.line_chart(frame[['Close', 'EMA50','EMA200']])
            except Exception as e:
                st.write(name,e)
        elif option1 == 'Sell'and option2 == 'EMA':   
            try:     
                if len(frame)>30 and len(framew)>30 and framew['Decision EMA50'].iloc[-1]=='Sell' \
                and frame['ADX'].iloc[-1]>=adx_value and (frame['MACD_diff'].iloc[-1]<0 or frame['Trend MACD'].iloc[-1]=='Strong')  \
                and (framew['MACD_diff'].iloc[-1]<0 or framew['Trend MACD'].iloc[-1]=='Strong') and frame['Decision EMA50_cross'].iloc[-1]=='Sell' \    
                and framew['sup'].iloc[-1]=='-1':    
                    sira +=1
                    st.write(str(sira)+" Selling EMA50 for "+ name)
                    st.write(frame.tail(2))
            except Exception as e:
                st.write(name,e)
        elif option1 == 'Buy'and option2 == 'MACD':  
            try:   
                if  len(frame)>30 and len(framew)>30 and frame['Decision MACD'].iloc[-1]=='Buy'  \
                and framew['Decision EMA50'].iloc[-1]=='Buy' and frame['ADX'].iloc[-1]>=adx_value\
                and (framew['MACD_diff'].iloc[-1]>0 or framew['Trend MACD'].iloc[-1]=='Strong') and framew['sup'].iloc[-1]=='1':    
                    sira +=1
                    st.write(str(sira)+" Buying Signal MACD/EMA200 for "+ name)
                    st.write(frame.tail(2))
            except Exception as e:
                st.write(name,e) 
        elif option1 == 'Sell'and option2 == 'MACD': 
            try: 
                if len(frame)>30 and len(framew)>30 and frame['Decision MACD'].iloc[-1]=='Sell'  \
                    and framew['Decision EMA50'].iloc[-1]=='Sell' and frame['ADX'].iloc[-1]>=adx_value \
                    and (framew['MACD_diff'].iloc[-1]<0 or framew['Trend MACD'].iloc[-1]=='Strong') and framew['sup'].iloc[-1]=='-1':
                        sira +=1
                        st.write(str(sira)+" Selling Signal MACD/EMA200 for "+ name)
                        st.write(frame.tail(2))
            except Exception as e:
                st.write(name,e)
        # elif option1 == 'Buy'and option2 == 'SUPERTREND':
            # try:
                # if len(frame)>30 and len(framew)>30 and frame['Decision Super'].iloc[-1]=='Buy' and framew['Decision EMA50'].iloc[-1]=='Buy' \
                # and frame['ADX'].iloc[-1]>=adx_value and (frame['MACD_diff'].iloc[-1]>0 or frame['Trend MACD'].iloc[-1]=='Strong')  \
                # and (framew['MACD_diff'].iloc[-1]>0 or framew['Trend MACD'].iloc[-1]=='Strong'):
                    # sira +=1
                    # st.write(str(sira)+" Buy Supertrend for "+ name)
                    # st.write(frame.tail(2))
            # except Exception as e:
                # st.write(name,e) 
        # elif option1 == 'Sell'and option2 == 'SUPERTREND':        
            # try:    
                # if len(frame)>30 and len(framew)>30 and frame['Decision Super'].iloc[-1]=='Sell' and framew['Decision EMA50'].iloc[-1]=='Sell' \
                # and frame['ADX'].iloc[-1]>=adx_value and (frame['MACD_diff'].iloc[-1]<0 or frame['Trend MACD'].iloc[-1]=='Strong') \
                # and (framew['MACD_diff'].iloc[-1]<0 or framew['Trend MACD'].iloc[-1]=='Strong'):
                    # sira +=1
                    # st.write(str(sira)+" Sell Supertrend for "+ name)
                    # st.write(frame.tail(2))
            # except Exception as e:
                # st.write(name,e)  
###HAFTALIK#####    
    # elif option3=='Week':
        # if option1 == 'Buy'and option2 == 'MACD':  
            # try:   
                # if  len(framew)>30 and framew['Decision MACD'].iloc[-1]=='Buy' and framew['MACD'].iloc[-1]<0 \
                # and framew['Decision EMA50'].iloc[-1]=='Buy': #and framew['ADX'].iloc[-1]>=adx_value:
                    # sira +=1
                    # st.write(str(sira)+" Buying Signal MACD/EMA200 for "+ name)
                    # st.write(framew.tail(2))
            # except Exception as e:
                # st.write(name,e) 
        # elif option1 == 'Sell'and option2 == 'MACD': 
            # try: 
                # if len(framew)>30 and framew['Decision MACD'].iloc[-1]=='Sell' and framew['MACD'].iloc[-1]>0 \
                    # and framew['Decision EMA50'].iloc[-1]=='Sell' : #and framew['ADX'].iloc[-1]>=adx_value :
                        # sira +=1
                        # st.write(str(sira)+" Selling Signal MACD/EMA200 for "+ name)
                        # st.write(framew.tail(2))
            # except Exception as e:
                # st.write(name,e)

        # if option1 == 'Buy'and option2 == 'SUPERTREND':
            # try:
                # if len(framew)>30 and framew['Decision Super'].iloc[-1]=='Buy' \
                # and (framew['MACD_diff'].iloc[-1]>0 or framew['Trend MACD'].iloc[-1]=='Strong') :
                    # sira +=1
                    # st.write(str(sira)+" Buy Supertrend for "+ name)
                    # st.write(framew.tail(2))
            # except Exception as e:
                # st.write(name,e) 
        # elif option1 == 'Sell'and option2 == 'SUPERTREND':        
            # try:    
                # if len(framew)>30 and framew['Decision Super'].iloc[-1]=='Sell'  \
                # and (framew['MACD_diff'].iloc[-1]<0 or framew['Trend MACD'].iloc[-1]=='Strong') :
                    # sira +=1
                    # st.write(str(sira)+" Sell Supertrend for "+ name)
                    # st.write(framew.tail(2))
            # except Exception as e:
                # st.write(name,e)  
        # if option1 == 'Buy'and option2 == 'EMA':  
            # try:
                # if len(framew)>30 and (framew['MACD_diff'].iloc[-1]>0 \
                # or framew['Trend MACD'].iloc[-1]=='Strong')and framew['Decision EMA50_cross'].iloc[-1]=='Buy': 
                    # sira +=1
                    # st.write(str(sira)+" Buying EMA50 for "+ name)
                    # st.write(framew.tail(2))
            # except Exception as e:
                # st.write(name,e)
        # elif option1 == 'Sell'and option2 == 'EMA':   
            # try:     
                # if len(framew)>30 and (framew['MACD_diff'].iloc[-1]<0 or framew['Trend MACD'].iloc[-1]=='Strong')  \
                # and framew['Decision EMA50_cross'].iloc[-1]=='Sell' :    
                    # sira +=1
                    # st.write(str(sira)+" Selling EMA50 for "+ name)
                    # st.write(framew.tail(2))
            # except Exception as e:
                # st.write(name,e)
