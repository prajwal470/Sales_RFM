#!/usr/bin/env python
# coding: utf-8

# # Introduction

# ### In a world where businesses are growing tremendously, and cater to a large number of customers on a regular basis. It becomes very essential for businesses to categorize their customers. Customer segmentation is an effective tool for businesses to closely align their strategy and tactics with, and better target, their customers. Every customer is different and every customer journey is different so a single approach often isn’t going to work for all. This is where customer segmentation becomes a valuable process.

# # Problem Description

# ### To bulid RFM Value and categorize the customers for given Data set

# ##  This project has been completed in 5 steps :-
# ## 1. Data Cleaning
# ## 2. Exploratory Data Analysis (EDA)
# ## 3. Data Transformation
# ## 4. Clustering and segmentation
# ## 5. Data visulization
# 

# # 1) Importing the data and Cleaning

# In[1]:


# Import Dependencies
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
import missingno
import seaborn as sns
import datetime as dt
import numpy as np
import squarify
import plotly.express as px
import regex as re
import plotly.graph_objects as go


# 
# here we will read our data and go through the columns and rows for further development
# 

# In[2]:


df = pd.read_csv('sales_data.csv')


# In[3]:


df.head(10)


# In[4]:


df.columns


# In[5]:


df.shape


# In[6]:


df.dtypes


# Different type of datatype in data set

# ## Data Cleaning and handleing for missing value
# * It is very important to clean the data and find the missing values as it may affect our model 
# * I am using an lib called missingo to verify the miising values in data sets
# * It returns a graph as we can easily visualise the data set
# 

# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


missingno.matrix(df,figsize=(30,10))


# * ### hear we can see the data set is clean and their is no missing value 

# # 2) EDA 
# 
# In EDA we are trying to figure out more about the data so you can build a model the best way you can.we usually do this when you first look at a dataset but it'll continually happen as you learn more. EDA is an iterative process. There's no one way to do it either. It'll vary with each new dataset

# * the basic idea is
# * 1) to make our RFM model based on customer Data
# * 2) to Know on which day, week of month the customer is active so we can give special deals
# * 3) this data set is clean and do not have any typo error

# ###  1)Top ten highest revenue 

# In[10]:


top_revenue = df['REVENUE'].sort_values(ascending = False)[:10]


# In[11]:


plt.figure(figsize = (10,5))
sns.barplot(x = [1,2,3,4,5,6,7,8,9,10], y = top_revenue)
plt.xlabel('Rank')
plt.ylabel('Income value')
plt.title('Top 10 highest revenue ')
plt.tight_layout()


# ### 2) Top ten highest orders

# In[12]:


top_orders = df['TOTAL_ORDERS'].sort_values(ascending = False)[:10]
plt.figure(figsize = (10,5))
sns.barplot(x = [1,2,3,4,5,6,7,8,9,10], y = top_orders)
plt.xlabel('Rank')
plt.ylabel('Orders Count')
plt.title('Top 10 orders ')
plt.tight_layout()


# ## 3) Days of week

# 
# ### weekly purchaces

# In[13]:


df_dates = df[['MONDAY_ORDERS', 'TUESDAY_ORDERS', 'WEDNESDAY_ORDERS',
       'THURSDAY_ORDERS', 'FRIDAY_ORDERS', 'SATURDAY_ORDERS', 'SUNDAY_ORDERS',
       'MONDAY_REVENUE', 'TUESDAY_REVENUE', 'WEDNESDAY_REVENUE',
       'THURSDAY_REVENUE', 'FRIDAY_REVENUE', 'SATURDAY_REVENUE',
       'SUNDAY_REVENUE']]


# In[14]:


df_dates.head()


# In[15]:


sum_of_orders = df_dates[['MONDAY_ORDERS', 'TUESDAY_ORDERS', 'WEDNESDAY_ORDERS',
       'THURSDAY_ORDERS', 'FRIDAY_ORDERS', 'SATURDAY_ORDERS', 'SUNDAY_ORDERS',
       'MONDAY_REVENUE', 'TUESDAY_REVENUE', 'WEDNESDAY_REVENUE',
       'THURSDAY_REVENUE', 'FRIDAY_REVENUE', 'SATURDAY_REVENUE',
       'SUNDAY_REVENUE']].sum()


# In[16]:


sum_of_orders = pd.DataFrame(sum_of_orders)


# ### Orders

# In[17]:


# making number of order on any week day 
orders_day = pd.DataFrame(sum_of_orders[:7] )
orders_day.reset_index(inplace = True)
orders_day.rename(columns = {'index':'Day',0:'NO_of_Orders'},inplace = True)
# removing "_"orders to make everything look clean
orders_day['Day']=orders_day['Day'].str.split('_').str[0]


# In[18]:


plt.figure(figsize = (10,5))
sns.barplot(x = orders_day['Day'], y = orders_day['NO_of_Orders'])
plt.xlabel('week')
plt.ylabel('Orders Count')
plt.title('orders on each week day')
plt.tight_layout()


# ### Revenue

# In[19]:


# making total revenue on any week day 
revenue_day = pd.DataFrame(sum_of_orders[7:])
revenue_day.reset_index(inplace = True)
revenue_day.rename(columns = {'index':'Day',0:'Sum_of_Revenue'},inplace = True)


# In[20]:


# removing "_"orders to make everything look clean
revenue_day['Day']=revenue_day['Day'].str.split('_').str[0]


# In[21]:


plt.figure(figsize = (10,5))
sns.barplot(x = revenue_day['Day'], y = revenue_day['Sum_of_Revenue'])
plt.xlabel('Week-Days')
plt.ylabel('Revenue in Million')
plt.title('Revenue on each days')
plt.tight_layout()


# ## From the plot for Weekly purchases it is seen that most of the purchases, occur in the Thursday Followed by Sunday and friday
# 

# * #### now we got orders_day and revenue_day

# In[22]:


#combain to make it one df
revenue_day['NO_of_Orders'] = orders_day['NO_of_Orders']


# In[23]:


# normalizing the revenue to make easy plot as revenue is in millions so dividing by 100 we will
# get in 10k range
revenue_day['Sum_of_Revenue_in_100'] = revenue_day['Sum_of_Revenue']/100


# ## 4) Week's in Month

# ### total months can be calucalted form first_order and last order 

# In[24]:


df_Fday = pd.to_datetime(df['FIRST_ORDER_DATE'].min())
df_Lday = pd.to_datetime(df['LATEST_ORDER_DATE'].max())


# In[25]:


# Finding to days years and months
total_days = (df_Lday-df_Fday).days
total_Year = total_days/365
total_month = total_Year*12


# In[26]:


df_months = df[['WEEK1_DAY01_DAY07_ORDERS',
       'WEEK2_DAY08_DAY15_ORDERS', 'WEEK3_DAY16_DAY23_ORDERS',
       'WEEK4_DAY24_DAY31_ORDERS', 'WEEK1_DAY01_DAY07_REVENUE',
       'WEEK2_DAY08_DAY15_REVENUE', 'WEEK3_DAY16_DAY23_REVENUE',
       'WEEK4_DAY24_DAY31_REVENUE']]
rename =['WEEK1_ORDERS',
       'WEEK2_ORDERS', 'WEEK3_ORDERS',
       'WEEK4_ORDERS', 'WEEK1_REVENUE',
       'WEEK2_REVENUE', 'WEEK3_REVENUE',
       'WEEK4_REVENUE']
columns = ['WEEK1_DAY01_DAY07_ORDERS', 'WEEK2_DAY08_DAY15_ORDERS',
       'WEEK3_DAY16_DAY23_ORDERS', 'WEEK4_DAY24_DAY31_ORDERS',
       'WEEK1_DAY01_DAY07_REVENUE', 'WEEK2_DAY08_DAY15_REVENUE',
       'WEEK3_DAY16_DAY23_REVENUE', 'WEEK4_DAY24_DAY31_REVENUE']
df_months = pd.DataFrame(df_months)
# changing column names
k = {}
for i,(x,y) in enumerate(zip(columns,rename)):
    k[x] = y
df_months.rename(columns = k, inplace = True)       


# In[27]:


df_months


# In[28]:


df_months_ = df_months[['WEEK1_ORDERS',
       'WEEK2_ORDERS', 'WEEK3_ORDERS',
       'WEEK4_ORDERS', 'WEEK1_REVENUE',
       'WEEK2_REVENUE', 'WEEK3_REVENUE',
       'WEEK4_REVENUE']].sum()
df_months_=pd.DataFrame(df_months_)


# ### Orders

# In[29]:


#month order
df_months_order = pd.DataFrame(df_months_[:4])
df_months_order.reset_index(inplace = True)
df_months_order.rename(columns = {'index':'Week_Num',0:'NUM_of_Orders'} , inplace = True)
df_months_order


# In[30]:


plt.figure(figsize = (10,5))
sns.barplot(x = df_months_order['Week_Num'], y = df_months_order['NUM_of_Orders'])
plt.xlabel('Week-Days')
plt.ylabel('Orders')
plt.title('Orders on each Weekly')
plt.tight_layout()


# ### Revenue

# In[31]:



df_months_revenue = pd.DataFrame(df_months_[4:])
df_months_revenue.reset_index(inplace = True)
df_months_revenue.rename(columns = {'index':'Week_Num',0:'Revenue'} , inplace = True)
df_months_revenue


# In[32]:


plt.figure(figsize = (10,5))
sns.barplot(x = df_months_revenue['Week_Num'], y = df_months_revenue['Revenue'])
plt.xlabel('Week-Days')
plt.ylabel('Revenue in Million')
plt.title('Revenue on each week')
plt.tight_layout()


# ## From the Above plot's. it is seen that most of the purchases, occur in the Week-4 
# 

# ## 4) Time of Day

# In[33]:


df_time = pd.DataFrame(df[['TIME_0000_0600_ORDERS',
       'TIME_0601_1200_ORDERS', 'TIME_1200_1800_ORDERS',
       'TIME_1801_2359_ORDERS', 'TIME_0000_0600_REVENUE',
       'TIME_0601_1200_REVENUE', 'TIME_1200_1800_REVENUE',
       'TIME_1801_2359_REVENUE']])
df_time_ = df_time[['TIME_0000_0600_ORDERS',
       'TIME_0601_1200_ORDERS', 'TIME_1200_1800_ORDERS',
       'TIME_1801_2359_ORDERS', 'TIME_0000_0600_REVENUE',
       'TIME_0601_1200_REVENUE', 'TIME_1200_1800_REVENUE',
       'TIME_1801_2359_REVENUE']].sum()
df_time_ = pd.DataFrame(df_time_)


# ### Orders

# In[34]:


df_Time_order = pd.DataFrame(df_time_[:4])
df_Time_order.reset_index(inplace = True)
df_Time_order.rename(columns = {'index':'Timings',0:'NUM_of_Orders'} , inplace = True)
df_Time_order


# In[35]:


plt.figure(figsize = (10,5))
sns.barplot(x = df_Time_order['Timings'][:10], y = df_Time_order['NUM_of_Orders'][:10])
plt.xlabel('Timings')
plt.ylabel('Orders')
plt.title('Timings of Day')
plt.tight_layout()


# ### Revenue

# In[36]:


df_Time_revenue = pd.DataFrame(df_time_[4:])
df_Time_revenue.reset_index(inplace = True)
df_Time_revenue.rename(columns = {'index':'Timings',0:'Revenue'} , inplace = True)
df_Time_revenue 


# In[37]:


plt.figure(figsize = (10,5))
sns.barplot(x = df_Time_revenue['Timings'], y = df_Time_revenue['Revenue'])
plt.xlabel('Timings')
plt.ylabel('Orders')
plt.title('Timings of Day')
plt.tight_layout()


# In[ ]:





# ## 5) Customer with high revenue and high order

# In[38]:


df_customer = pd.DataFrame(df[['CustomerID','REVENUE','TOTAL_ORDERS']])
df_customer_revenue = pd.DataFrame(df[['CustomerID','REVENUE']])
df_customer_orders = pd.DataFrame(df[['CustomerID','TOTAL_ORDERS']])


# ### Orders

# In[39]:


df_customer_orders .sort_values(by =['TOTAL_ORDERS'],ascending=False,inplace = True)
df_customer_orders.reset_index(inplace = True)
df_customer_orders.drop(columns=['index'],inplace =True)
df_customer_orders.head(10)


# ### Revenue

# In[40]:


df_customer_revenue.sort_values(by =['REVENUE'],ascending=False,inplace = True)
df_customer_revenue.reset_index(inplace = True)
df_customer_revenue.drop(columns=['index'],inplace =True)
df_customer_revenue.head(10)


# ### From above tables CustomerID 1  Has high Revenue and CustomerID 26 has high Order list

# ## 6) Correlation between variables

# In[41]:


k = df.columns[1:24]
df1 = pd.DataFrame(df[k])


# In[42]:


plt.figure(figsize = (18,12))
sns.heatmap(df1.corr(), annot = True)
plt.title('CORRELTAION MATRIX')


# ### From the correaltion matrix, it is understood that most columns are not correlated to each other. Except for Day and weeks, they are highly correlated . Where as 'AVGdays' and 'Days since last order' are negatively correlated.

# # EDA Summary
# 

# ### From Above EDA Process we can assume that

# * 156 is the Highest Ordes from a single person
# * 34847 is the highest revenue from a single person
# * Thursday and Sunday are the Highest in revenue and Order placed
# * most Shopping happens at month End 
# * some People have less order but average cost of each item is high
# * If person purchaced more than 3 times he tend's to shop more

# # 2) Data Transformation 
# ### Performing RFM Segmentation and RFM Analysis

# ## RFM model 

# ### * The idea is to divide the customer based on their Recency , Frequency , Monetary
# ### * RFM model will be the best fit for this data
# * Recency: How much time has elapsed since a customer’s last activity or transaction with the brand
# * Frequency: How often has a customer transacted or interacted with the brand during a particular period of time
# * Monetary: Also referred to as “monetary value,” this factor reflects how much a customer has spent with the brand during a particular period of time.

# ### 1) Recency

# Recency factor is based on the notion that the more recently a customer has made a purchase with a company, the more likely they will continue to keep the business and brand in mind for subsequent purchases. This information can be used to remind recent customers to revisit the business soon to continue meeting their purchase needs.

# now we calculate recency using our data set

# In[43]:


df.head()


# In[44]:


# convert to date time
df['Date']= pd.to_datetime(df['LATEST_ORDER_DATE'])


# In[45]:


# grouping the based on customerID and laste date of order
df_RFM = df.groupby(by='CustomerID',as_index=False)['Date'].max()


# In[46]:


# recent date will be the latest order amoung all orders
recent_date = df_RFM['Date'].max()


# In[47]:


# as defination Recency is latest order date - last date of that customer ordered
df_RFM['Recency'] = df_RFM['Date'].apply(lambda x: (recent_date - x).days)


# In[48]:


df_RFM.head()


# ### 2) Frequency
# The frequency of a customer’s transactions may be affected by factors such as the type of product, the price point for the purchase, and the need for replenishment or replacement. Predicting this can assist marketing efforts directed at reminding the customer to visit the business again.
# 
# 

# alredy we have total-order from each customer so the frequency will be easy to find

# In[49]:


df_RFM['Frequency'] = df['TOTAL_ORDERS']


# ### 3) Monetary Value
# Monetary value stems from how much the customer spends. A natural inclination is to put more emphasis on encouraging customers who spend the most money to continue to do so. While this can produce a better return on investment in marketing and customer service, it also runs the risk of alienating customers who have been consistent but may not spend as much with each transaction.
# 
# 
# * in this data set we have carriage revenue 
# * we should minus this value from total revenue of  customer to get our exact profit

# In[50]:


df_RFM['Monetary'] = df['REVENUE'] - df['CARRIAGE_REVENUE']


# In[51]:


df_RFM.drop('Date', inplace=True, axis=1)


# In[52]:


df_RFM.head()


# In[53]:


plt.figure(figsize=(12,10))
# Plot distribution of R
plt.subplot(3, 1, 1); sns.distplot(df_RFM['Recency'])
# Plot distribution of F
plt.subplot(3, 1, 2); sns.distplot(df_RFM['Frequency'])
# Plot distribution of M
plt.subplot(3, 1, 3); sns.distplot(df_RFM['Monetary'])
# Show the plot
plt.show()


# In[54]:


sns.pairplot(df_RFM[['Recency','Frequency','Monetary']])


# ## By Analysing both plots we can say that
# 

# * we can see that in recency, that we have some regulars who are buying frequently as some customer who are we loseing at starting of graph and ending
# * and we can see that higher the frequency higher is the revenue from customers
# * There are many recent purchaces with higher monetary value than older purchases.
# * Frequency and monetary variables have slight linear trend.

# ### There are some customers who are potential outliers, but these cannot be removed because, for example there is a customerID 1 have high revenue but less order compared to customerID26. He could be vital to the business. There is also another customer who has frequently billed a high value. Hence, if these are removed, business could miss classifying their main customers, who could potentially be of high value in the future also.

# In[55]:


from sklearn.preprocessing import StandardScaler, Normalizer
rfm_df_copy = df_RFM.copy()
rfm_df_copy.set_index('CustomerID', inplace= True)


# In[56]:


scaler = StandardScaler()
normal = Normalizer()
scaled_data = scaler.fit_transform(rfm_df_copy)
scaled_data = normal.fit_transform(scaled_data)
rfm_scaled = pd.DataFrame(scaled_data, columns = ['Recency','Frequency','Monetary'])
rfm_scaled.set_index(rfm_df_copy.index, inplace=True)


# In[57]:


rfm_scaled.describe()


# # 4) Clustering

# In[58]:


from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.metrics import silhouette_score


# In[59]:


# Creating an kmeans model
kmeans = KMeans()


# ### KMeans requires the number of clusters to be specified during the model building process. To know the right number of clusters, we use elbow method and silhouette analysis to get the number of optimal clusters

# In[60]:


def elbow_method(X):
  
  
    metrics = ['distortion', 'calinski_harabasz', 'silhouette']
  
    for m in metrics:
        visualizer = KElbowVisualizer(kmeans, k = (2,10), metric = m)
        visualizer.fit(X)
        visualizer.poof()


# In[61]:


# Using the elbow method function to understand optimum number of clusters
elbow_method(rfm_scaled)


# ## From the elbow method  it is clearly understood that, 4 clusters is performing the best. Hence, 4 clusters will be selected to build the KMeans model and classify the customers.

# In[62]:


kmeans = KMeans(n_clusters = 4, random_state=10)


# In[63]:


kmeans.fit(rfm_scaled)


# In[64]:


labels = kmeans.predict(rfm_scaled)
rfm_df_copy['Cluster'] = labels
rfm_df_copy.head(10)


# In[65]:


plt.figure(figsize = (18,10))
sns.scatterplot(x = rfm_df_copy['Recency'], y = rfm_df_copy['Frequency'], size= rfm_df_copy['Monetary'], hue = rfm_df_copy['Cluster'])


# ### Making groups of Recency, Frequency, Monetary and Calculating Score

# In[66]:


# Grouping by clusters to understand the profiles
rfm_df_copy.groupby('Cluster').mean()


# In[67]:


# Number of customers belonging to each cluster
rfm_df_copy['Cluster'].value_counts()


# # 5) Segmentation
# ## potential customer segmentation using RFM model and some meaningful insights from each segment.

# 
# ### Score 10 and above are "Champions" and there are in top 25%
# ### Score 8 and above but below 10 are "Loyal" and there are in top 50%
# ### Score 5 and above but below 8 are "Potential customers" and there are in top 75%
# ### Score 4 and above but below 5 are "promising customers"
# ### Score below 4 are Requires Attention
# ### Champions and  belongs to cluster 3
# ### Loyal customers belongs to cluster 2
# ### Potential customers  belongs to cluster 1
# ### Requires Attention belongs to cluster 0

# In[68]:


df_RFM.head()


# In[69]:


#Calculating R and F groups
# Create labels for Recency and Frequency
r_labels = range(4, 0, -1); 
f_labels = range(1, 5)
# Assign these labels to 4 equal percentile groups 
r_groups = pd.qcut(df_RFM['Recency'], q=4, labels=r_labels)
# Assign these labels to 4 equal percentile groups 
f_groups = pd.qcut(df_RFM['Frequency'], q=4, labels=f_labels)
# Create new columns R and F 
df_RFM = df_RFM.assign(R = r_groups.values, F = f_groups.values)
# Create labels for Monetary
m_labels = range(1, 5)
# Assign these labels to three equal percentile groups 
m_groups = pd.qcut(df_RFM['Monetary'], q=4, labels=m_labels)
# Create new column M
df_RFM = df_RFM.assign(M = m_groups.values)
df_RFM.head()


# In[70]:


def join_rfm(x): return x['R'] + x['F'] +x['M']
df_RFM['Score'] = df_RFM.apply(join_rfm, axis=1)
df_RFM.head()


# In[71]:


def customer_level(df):
    if df['Score'] >= 10:
        return 'Champions'
    elif ((df['Score'] >= 8) and (df['Score'] < 10)):
        return 'Loyal_customers'
    elif ((df['Score'] >= 5) and (df['Score'] < 8)):
        return 'Potential_customers'
    elif ((df['Score'] >= 0) and (df['Score'] < 5)):
        return 'Requires Attention'


# In[72]:


df_RFM['Score_level'] = df_RFM.apply(customer_level,axis = 1)


# In[73]:


df_RFM.head()


# In[74]:


df['RFM_Score'] = df_RFM['Score']


# In[75]:


RFM_level_agg = df_RFM.groupby('Score_level').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']
}).round(1)
# Print the aggregated dataset
print(RFM_level_agg)


# ### I made some data adjustment for data visulization

# # 6) Data visulization
# 

# ### Customer Level 

# In[76]:


RFM_level_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
#Create our plot and resize it.
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(16, 9)
squarify.plot(sizes=RFM_level_agg['Count'], 
              label=[
                     'Champions',
                     'Loyal_customers',
                     'Potential_customers',
                      'Promising_custommers', 
                       'Requires Attention'], alpha=1,color=plt.cm.Set2.colors )
plt.title("RFM Segments",fontsize=18,fontweight="bold")
plt.axis('off')
plt.show()


# ### Pie chart of customer level

# In[77]:


count = df_RFM.Score_level.value_counts()
name = df_RFM.Score_level.value_counts().index
fig = px.pie(df_RFM, values= count, names=name,
             title='Customer level segments',
              labels=name)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


# ### Trade of B/W Orders and Revenue

# In[78]:


days = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY',
       'SUNDAY']
fig = go.Figure()
fig.add_trace(go.Bar(x=days,
                y=revenue_day['NO_of_Orders'],
                name='Orders',
                marker_color='rgb(55, 83, 109)'
                ))
fig.add_trace(go.Bar(x=days,
                y=revenue_day['Sum_of_Revenue_in_100'],
                name='Revenue',
                marker_color='rgb(26, 118, 255)'
                ))

fig.update_layout(
    title='Trade of B/W Orders and Revenue',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='Revenue * 100 for original value',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
)
fig.show()


# * by observing the graph we can say that on monday we have less revenue than whole week
# * Thursday has high Revenue  and Orders of the whole week
# * on Saturday We have low orders
# 

# ### Combianed pie chart of revenue and customer level

# In[79]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots

labels = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY',
       'SUNDAY']

# Create subplots: use 'domain' type for Pie subplot
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=labels, values=revenue_day['Sum_of_Revenue_in_100'], name="revenue"),
              1, 1)
fig.add_trace(go.Pie(labels=name, values=count, name="Custormers"),
              1, 2)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.4, hoverinfo="label+percent+name")

fig.update_layout(
    title_text="Combianed pie chart of revenue and customer level",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='DAYS', x=0.183, y=0.5, font_size=18, showarrow=False),
                 dict(text='LEVEL', x=0.82, y=0.5, font_size=18, showarrow=False)])
fig.show()


# # Conclusion

# ### In this project, a translational dataset online store was used. The data set contained various columns. It contains data for almost a period of 7 year. The main aim of the project was to classify the customers into different segments. These segments will have a defining character of their own. This will help the business cater better to their customers which inturn could increase the profits.

# ## 1) data Cleaning :- the data set was clean we given
# 
# ## 2) Exploratory Data Analysis (EDA)
# 
#   ### *156 is the Highest Ordes from a single person
#   ### * 34847 is the highest revenue from a single person
#   ### * Thursday and Sunday are the Highest in revenue and Order placed
#   ### * most Shopping happens at month End
#   ### * some People have less order but average cost of each item is high
#   ### * If person purchaced more than 3 times he tend's to shop more
# 
# ## 3) Data Transformation 
#   ### * In this section, a Recency, Frequency and Monetary analysis Model was developed for each customerID
# 
# ## 4) Clustering and 5) segmentation
# 
# ###  * In this section, the optimum number of clusters were chosen via elbow  method It was found that 4 clusters would be the most optimum. 
# ### * A KMeans model with 4 clusters was developed. 
# ### * Each customer ID was clustered into one of the 4 clusters. and named based on their Score Champions ,  Loyal , Potential , Requires Attention 
# 
# ## 6) Data Data visulization :- Ploted some pie chart and barchart for further analysis
# 
# 

# ## * On the basis of this analysis, the business can offer attractive deals to its Potential and low value customers and they can also treat their high value customers with special business offers such as loyalty points.
# ## * they can even make spechial day on weekday's as monday to boost their revenue on monday

# In[ ]:





# In[ ]:





# In[ ]:




