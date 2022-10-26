import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
#import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

# -----Title of  the dashborad
st.title('Default Rate and Macroeconomic Indicators')

# -----Read CSV or Excel file and load data
#@st.cache
def load_data():
    # Read Excel
    excel_file= 'HW6_DATA.xlsx'
    sheet_name='Sheet1'
    df = pd.read_excel(excel_file,sheet_name=sheet_name,usecols='A:N',header=0)
    # Read CSV
    #df = pd.read_table(r'C:\Users\user\Documents\JupyterWork\CMSE830_FDS\HW5_data\HW5_DATA_FINAL.csv',sep=',')    
    # ----Convert YYYYMMDD dtype
    df['YYYYMMDD'] = df['YYYYMMDD'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
    df['YYYY-MM'] = pd.DatetimeIndex(df['YYYYMMDD']).strftime('%Y-%m')
    df = df.drop('YYYYMMDD', axis=1)
    first_column = df.pop('YYYY-MM')
    df.insert(0, 'YYYY-MM', first_column)
    # -----Set index
    df = df.set_index('YYYY-MM')
    return df

data=load_data()

# -----Scaling data using the sklearn.preprocessing.MinMaxScaler()
scaler = MinMaxScaler()
data[["SCALED_DEFAULT_RATE","SCALED_S&P","SCALED_NASDAQ","SCALED_CPI","SCALED_PPI","SCALED_MORTGAGE_RATE","SCALED_UNEMPLOYMENT_RATE","SCALED_INFLATION_RATE","SCALED_DISPOSABLE_INCOME","SCALED_QUARTERLY_REAL_GDP","SCALED_CORP_BONDYIELD_RATE","SCALED_IMPORT_PRICE_INDEX"]] = scaler.fit_transform(data[["DEFAULT_RATE","S&P","NASDAQ","CPI","PPI","MORTGAGE_RATE","UNEMPLOYMENT_RATE","INFLATION_RATE","DISPOSABLE_INCOME","QUARTERLY_REAL_GDP","CORP_BONDYIELD_RATE","IMPORT_PRICE_INDEX"]])

# -----Define select variables
original_columns = data[['DEFAULT_RATE','S&P','NASDAQ','CPI','PPI','MORTGAGE_RATE','UNEMPLOYMENT_RATE','INFLATION_RATE','DISPOSABLE_INCOME','QUARTERLY_REAL_GDP','CORP_BONDYIELD_RATE','IMPORT_PRICE_INDEX']].columns 
scaled_columns = data[['SCALED_DEFAULT_RATE','SCALED_S&P','SCALED_NASDAQ','SCALED_CPI','SCALED_PPI','SCALED_MORTGAGE_RATE','SCALED_UNEMPLOYMENT_RATE','SCALED_INFLATION_RATE','SCALED_DISPOSABLE_INCOME','SCALED_QUARTERLY_REAL_GDP','SCALED_CORP_BONDYIELD_RATE','SCALED_IMPORT_PRICE_INDEX']].columns

# ------checkbox
checkbox_1 = st.checkbox("Reveal dataset of the Default Rate and Macroeconomic Indicators")

# -----Column Description
expander = st.expander("See Indicators' Description")
expander.write("""	**Annual Default Rate(%)** : Default Rate of global corporates for each year""")
expander.write("""  **S&P** : Average of the highest and lowest recorded price for the fisrt day of month""")
expander.write("""  **NASDAQ** : The NASDAQ Composite Index is a market capitalization weighted index with more than 3000 common equities listed on the NASDAQ Stock Market.""")
expander.write("""  **CPI** : The Consumer Price Index for All Urban Consumers""")
expander.write("""  **PPI** : Producers Purchase Index- Construction Materials""")
expander.write("""  **Mortgage Rate(%)** : 30-Year Fixed Rate Mortgage Average in the United States""")
expander.write("""  **Unemployment Rate(%)** : The unemployment rate represents the number of unemployed as a percentage of the labor force. Labor force data are restricted to people 16 years of age and older""")
expander.write("""  **Inflation Rate(%)** : Inflation rate in the US""")
expander.write("""  **Disposable Income($)** : Real personal disposable income""")
expander.write("""  **Quarterly Real GDP($)** : Real US GDP for each quarter""")
expander.write("""  **Corpation Bond Yield Rate(%)** : The rate of return an investor will realize on a bond """)
expander.write("""  **Import Price Index** : All imports, 1-month percent change, not seasonally adjusted""")

if checkbox_1:
    st.dataframe(data=data)

# -----Trend line - default rate data with macroeconomic indicators
st.subheader('Time-Series Chart')
st.caption('Explore historical trend of default rate and economic index. You can compare any variable ')

# -----Draw line graph with original columns
select_box1 = st.multiselect(label = "1. Original Index", options = original_columns)
st.caption('Select multi-variables under the Original Index.') 
data_original = data[select_box1]
fig_org = px.line(data_original)
fig_org.add_vrect(x0="2007-12",x1="2009-06",fillcolor="gray",opacity=0.30,line_width=0,
                annotation_text="Great<br>Recession", annotation_position="top left",
                annotation_font_size=11,annotation_font_color="black")
fig_org.add_vrect(x0="2020-02",x1="2020-04",fillcolor="gray",opacity=0.30,line_width=0,
                annotation_text="COVID-19", annotation_position="top left",
                annotation_font_size=11,annotation_font_color="black")                
fig_org.update_layout(autosize=False,width=700,height=500)
fig_org.update_layout(legend=dict(orientation="h",yanchor="bottom",y=-0.2,xanchor="right",x=1))
st.write(fig_org)

# -----Draw line graph with scaled columns
select_box2 = st.multiselect(label = "2. Scaled Index", options = scaled_columns)
st.caption('Select preprocessed default rate and economic index ranged from 0 to 1.')
data_scaled = data[select_box2]
fig_scl = px.line(data_scaled)
fig_scl.add_vrect(x0="2007-12",x1="2009-06",fillcolor="gray",opacity=0.30,line_width=0,
                annotation_text="Great<br>Recession", annotation_position="top left",
                annotation_font_size=11,annotation_font_color="black")
fig_scl.add_vrect(x0="2020-02",x1="2020-04",fillcolor="gray",opacity=0.30,line_width=0,
                annotation_text="COVID-19", annotation_position="top left",
                annotation_font_size=11,annotation_font_color="black")                
fig_scl.update_layout(autosize=False,width=700,height=500)
fig_scl.update_layout(legend=dict(orientation="h",yanchor="bottom",y=-0.2,xanchor="right",x=1)) 
st.write(fig_scl)


# -----Correlation betweeen Default Rate and Macroeconomic index
# ------Correlation values are same when using scaled indices!
st.subheader('Correlation Matrix and Chart')

checkbox_2 = st.checkbox("Reveal the correlation matrix")
data_scaled = data[['DEFAULT_RATE','S&P','NASDAQ','CPI','PPI','MORTGAGE_RATE','UNEMPLOYMENT_RATE','INFLATION_RATE','DISPOSABLE_INCOME','QUARTERLY_REAL_GDP','CORP_BONDYIELD_RATE','IMPORT_PRICE_INDEX']]
fig_cor, ax = plt.subplots(figsize=(12,5))
mask = np.triu(np.ones_like(data_scaled.corr()))
sns.color_palette("vlag", as_cmap=True)
sns.heatmap(data_scaled.corr(), ax=ax, annot=True, cmap="vlag_r",mask=mask)

if checkbox_2:
    st.write(fig_cor)

st.caption('Select one variable to check the relation between Default Rate and each variable.')    

col3, col4, col5 = st.columns(3)

with col3:
    select_box1 = st.selectbox(label = "X1 axis", options = original_columns)
    fig_3 = sns.lmplot(x=select_box1, y='DEFAULT_RATE', data=data, lowess=False, palette="Set1")
    plt.xlabel(select_box1, fontsize=17)
    plt.ylabel('Default Rate', fontsize=17)
    st.pyplot(fig_3)
    
    corr = data[['DEFAULT_RATE', select_box1]].corr()
    corr = np.round(corr,decimals=4)
    corr_val=corr.iat[0, 1]
    st.markdown('Correlation Value X1:')
    st.write(corr_val)

with col4:
    select_box1 = st.selectbox(label = "X2 axis", options = original_columns)
    fig_4 = sns.lmplot(x=select_box1, y='DEFAULT_RATE', data=data, lowess=False, palette="Set1")
    plt.xlabel(select_box1, fontsize=17)
    plt.ylabel(' ', fontsize=17)    
    st.pyplot(fig_4)

    corr = data[['DEFAULT_RATE', select_box1]].corr()
    corr = np.round(corr,decimals=4)
    corr_val=corr.iat[0, 1]
    st.markdown('Correlation Value X2:')
    st.write(corr_val)

with col5:
    select_box1 = st.selectbox(label = "X3 axis", options = original_columns)
    fig_5 = sns.lmplot(x=select_box1, y='DEFAULT_RATE', data=data, lowess=False, palette="Set1")
    plt.xlabel(select_box1, fontsize=17)
    plt.ylabel(' ', fontsize=17)    
    st.pyplot(fig_5)    

    corr = data[['DEFAULT_RATE', select_box1]].corr()
    corr = np.round(corr,decimals=4)
    corr_val=corr.iat[0, 1]
    st.markdown('Correlation Value X3:')
    st.write(corr_val)