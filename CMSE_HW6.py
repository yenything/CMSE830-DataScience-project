import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
#import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt

# -----Title of  the dashborad
st.title('Default Rate and Macroeconomic Indicators')

# -----Column Description
expander = st.expander("See Indicators' Descriptions")
expander.write("""	**DEFAULT_RATE** : Global Corporate Default Rate""")
expander.write("""  **S&P** : Average of the highest and lowest recorded price for the fisrt day of month""")
expander.write("""  **NASDAQ** : The NASDAQ Composite Index is a market capitalization weighted index with more than 3000 common equities listed on the NASDAQ Stock Market.""")
expander.write("""  **CPI** : The Consumer Price Index for All Urban Consumers""")
expander.write("""  **PPI** : Producers Purchase Index- Construction Materials""")
expander.write("""  **MORTGAGE_RATE** : 30-Year Fixed Rate Mortgage Average in the United States""")
expander.write("""  **UNEMPLOYMENT_RATE** : The unemployment rate represents the number of unemployed as a percentage of the labor force. Labor force data are restricted to people 16 years of age and older""")
expander.write("""  **INFLATION_RATE** : Inflation rate in the US""")
expander.write("""  **DISPOSABLE_INCOME** : Real personal disposable income""")
expander.write("""  **QUARTERLY_REAL_GDP** : DESCRIPTION""")
expander.write("""  **CORP_BONDYIELD_RATE** : DESCRIPTION""")
expander.write("""  **IMPORT_PRICE_INDEX** : All imports, 1-month percent change, not seasonally adjusted.""")

# -----Read CSV or Excel file and load data
#@st.cache
def load_data():
    # Read Excel
    excel_file= 'HW6_DATA.xlsx'
    sheet_name='Sheet1'
    df = pd.read_excel(excel_file,sheet_name=sheet_name,usecols='A:N',header=0)
    # Read CSV
    #df = pd.read_table(r'C:\Users\user\Documents\JupyterWork\CMSE830_FDS\HW5_data\HW5_DATA_FINAL.csv',sep=',')    
    # Convert YYYYMMDD dtype
    df['YYYYMMDD'] = df['YYYYMMDD'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
    # Set index
    df = df.set_index('YYYYMMDD')
    return df

data=load_data()

# -----Scaling data using the sklearn.preprocessing.MinMaxScaler()
scaler = MinMaxScaler()
data[["SCALED_DEFAULT_RATE","SCALED_S&P","SCALED_NASDAQ","SCALED_CPI","SCALED_PPI","SCALED_MORTGAGE_RATE","SCALED_UNEMPLOYMENT_RATE","SCALED_INFLATION_RATE","SCALED_DISPOSABLE_INCOME","SCALED_QUARTERLY_REAL_GDP","SCALED_CORP_BONDYIELD_RATE","SCALED_IMPORT_PRICE_INDEX"]] = scaler.fit_transform(data[["DEFAULT_RATE","S&P","NASDAQ","CPI","PPI","MORTGAGE_RATE","UNEMPLOYMENT_RATE","INFLATION_RATE","DISPOSABLE_INCOME","QUARTERLY_REAL_GDP","CORP_BONDYIELD_RATE","IMPORT_PRICE_INDEX"]])

# -----Define select variables
#numeric_columns = data.select_dtypes(['float64', 'float32', 'int64', 'int32']).columns
#category_columns = data.select_dtypes(['object']).columns 
original_columns = data[['DEFAULT_RATE','S&P','NASDAQ','CPI','PPI','MORTGAGE_RATE','UNEMPLOYMENT_RATE','INFLATION_RATE','DISPOSABLE_INCOME','QUARTERLY_REAL_GDP','CORP_BONDYIELD_RATE','IMPORT_PRICE_INDEX']].columns 
scaled_columns = data[['SCALED_DEFAULT_RATE','SCALED_S&P','SCALED_NASDAQ','SCALED_CPI','SCALED_PPI','SCALED_MORTGAGE_RATE','SCALED_UNEMPLOYMENT_RATE','SCALED_INFLATION_RATE','SCALED_DISPOSABLE_INCOME','SCALED_QUARTERLY_REAL_GDP','SCALED_CORP_BONDYIELD_RATE','SCALED_IMPORT_PRICE_INDEX']].columns

#checkbox widget
checkbox = st.sidebar.checkbox("Reveal data")

if checkbox:
    # st.write(data)
    st.dataframe(data=data)

# -----Trend line - default rate data with macroeconomic indicators
# -----Draw line graph with original columns
st.subheader('This is a subheader1')
st.caption('This is a string that explains something above.')
#select_box1 = st.sidebar.multiselect(label = "Original Index", options = original_columns)
select_box1 = st.multiselect(label = "Original Index", options = original_columns)
data_original = data[select_box1]
fig_org = px.line(data_original)
st.write(fig_org)


# -----Draw line graph with scaled columns
st.subheader('This is a subheader2')
st.caption('Think another scaling way.. maybe logit?')
select_box2 = st.multiselect(label = "Scaled Index", options = scaled_columns)
data_scaled = data[select_box2]
fig_scl = px.line(data_scaled)
st.write(fig_scl)


# -----Correlation betweeen Default Rate and Macroeconomic index
# Same correlation values when using scaled indices!
st.subheader('This is a subheader3')
st.caption('This is a string that explains something above.')
data_scaled = data[['DEFAULT_RATE','S&P','NASDAQ','CPI','PPI','MORTGAGE_RATE','UNEMPLOYMENT_RATE','INFLATION_RATE','DISPOSABLE_INCOME','QUARTERLY_REAL_GDP','CORP_BONDYIELD_RATE','IMPORT_PRICE_INDEX']]
fig_33, ax = plt.subplots()
sns.color_palette("vlag", as_cmap=True)
sns.heatmap(data_scaled.corr(), ax=ax, annot=True, cmap="vlag")
st.write(fig_33)

# -----create scatterplots
# Same correlation values when using scaled indices!
st.subheader('This is a subheader4')
st.caption('This is a string that explains something above.')
st.sidebar.subheader("Scatter plot setup")
select_box1 = st.sidebar.selectbox(label = "X axis", options = original_columns)
select_box2 = st.sidebar.selectbox(label = "Y axis", options = original_columns)
# >>fig_1 = sns.relplot(x=select_box1, y=select_box2, hue = select_box7, data=data, kind="line")
fig_1 = sns.lmplot(x=select_box1, y=select_box2, data=data, lowess=False, palette="Set1",height=10, aspect=1, size=6)
st.pyplot(fig_1)

