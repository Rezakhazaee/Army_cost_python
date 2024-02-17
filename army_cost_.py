

import numpy as np 
import pandas as pd
import random
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.express as px
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')  # ignore notifications
np.set_printoptions(suppress=True)  # Removing the scientific record in numpy
pd.set_option('display.float_format', '{:.10g}'.format)  # Removing the scientific record in pandas

PATH = 'military_expenditure.csv'
df = pd.read_csv(PATH)

df

df.describe()  # statistical characteristics

print(f'Dataset size: {df.shape}')

print(f'Columns:\n {list(df.columns)}')

print(f'Data types: {df.dtypes}')

print(f'Number of missing values:\n{df.isna().sum()}')

countries = df.country.unique().tolist()
print(f'All countries:\n{countries}')

# We will remove the excess for convenienc of analysis
rf = df.loc[df['incomeLevel'] != 'Aggregates', :]

plt.figure(figsize=(20,8))
years = [i for i in range(df['year'].min(), df['year'].max() + 1)]
plt.grid()
world_military_expenditure = df[df['country'] == 'World']['Military expenditure (current USD)']
sns.lineplot(years, world_military_expenditure)
plt.title('Military spending around the World for 1980-2020', fontsize=18)
plt.xlabel('Year', fontsize=18)
plt.ylabel('Spending', fontsize=18)
plt.xticks(rotation=30, fontsize=18)
plt.yticks(fontsize=18)
plt.show()

max_min = rf.groupby('country').max()['Military expenditure (current USD)'] - rf.groupby('country').min()['Military expenditure (current USD)']
top_10 = max_min.nlargest(10)

plt.figure(figsize=(20, 8))
plt.title('Countries with the largest increase in military spending', fontsize=18)
plt.ylabel('Maximum spending - minimum spending', fontsize=18)
sns.barplot(top_10.index, top_10.values)
plt.xticks(rotation=30, fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Country', fontsize=18)
plt.show()

top_5 = rf.groupby('country').sum()['Military expenditure (current USD)'].nlargest(5)
top_5 = list(top_5.index)

def plotter_1(top_5):
    plt.figure(figsize=(20, 8))
    plt.title('Military expenditures of 5 countries (max values) for different times', fontsize=18)
    YEARS = [i for i in range(df['year'].min(), df['year'].max() + 1)]
    for top_country in top_5:
        EXPENDITURE = df.loc[df['country'] == top_country, 'Military expenditure (current USD)']
        sns.lineplot(YEARS, EXPENDITURE, label=top_country)
    plt.legend(fontsize=18)
    plt.xticks(rotation=30, fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('Year', fontsize=18)
    plt.ylabel('Military expenditure (current USD)', fontsize=18)
plotter_1(top_5)
plt.grid()   

tf = df.loc[df['Military expenditure (current USD)'] > 0,:]
smallest_5 = tf.groupby('country').sum()['Military expenditure (current USD)'].nsmallest(5)
smallest_5 = list(smallest_5.index)
def plotter_10(smallest_5):
    plt.figure(figsize=(20, 8))
    plt.title('5 countries with the lowest military spending for the period 1970-2020', fontsize=18)
    YEARS = [i for i in range(df['year'].min(), df['year'].max() + 1)]
    for top_country in smallest_5:
        EXPENDITURE = df.loc[df['country'] == top_country, 'Military expenditure (current USD)']
        sns.lineplot(YEARS, EXPENDITURE, label=top_country)
    plt.legend(fontsize=18)
    plt.xticks(rotation=30, fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('Year', fontsize=18)
    plt.ylabel('Military expenditure (current USD)', fontsize=18)
plotter_10(smallest_5)
plt.grid() 
 
def plotter_2(Income, n):
    fig, ax = plt.subplots(figsize=(20, 8))
    hf = df.loc[df['incomeLevel'] == Income, ['country','Military expenditure (% of GDP)']]
    random.choices(hf['country'].tolist(), k=n)
    hf = hf.loc[hf['country'].isin(random.choices(hf['country'].tolist(), k=n)), :]
    group = hf.groupby('country')
    max_Military_expenditure = group.max()['Military expenditure (% of GDP)']
    min_Military_expenditure = group.min()['Military expenditure (% of GDP)']
    ax.set_title(f'Analysis of {Income} countries', fontsize=17)
    width = 0.3
    x = np.arange(len(max_Military_expenditure.index.tolist()))
    ax.bar(x - width/2, max_Military_expenditure.values, width=0.3, label='Maximum % of GPD')
    ax.bar(x + width/2, min_Military_expenditure.values, width=0.3, label='Maximum % of GPD')
    ax.set_xticks(x)
    ax.set_xticklabels(max_Military_expenditure.index.tolist(), rotation=90, fontsize=13)
    plt.grid()
    ax.legend(fontsize=15, loc=1)
    ax.set_xlabel('Country', fontsize=16)
    ax.set_ylabel('Military expenditure (% of GDP)', fontsize=16)
    plt.yticks(fontsize=15)
    plt.show()

plotter_2('High income', 70)  

plotter_2('Low income', 70)

income_level_list = df['incomeLevel'].dropna().unique().tolist()
income_level_list.remove('Not classified')
income_level_list.remove('Aggregates')
income_level_list

ef = pd.DataFrame(df.groupby('incomeLevel').sum()['Military expenditure (current USD)'])
ef.drop(index=['Aggregates', 'Not classified'], inplace=True)
explode = (0.05, 0.05, 0.05, 0.05)
plt.figure(figsize=(10, 10))
plt.title('''The ratio of the total military spending of
countries with different levels of income of the population for the period 1970-2020''', fontsize=17)
plt.pie(ef['Military expenditure (current USD)'], labels=income_level_list, autopct='%1.1f%%', shadow=False,
        wedgeprops={'lw':1, 'ls':'--','edgecolor':'k'}, rotatelabels=True, explode=explode,
        colors = sns.color_palette('pastel')[0:4], textprops={'fontsize': 14})
plt.show()

df_update = df[df['country'].isin(['Low income', 'Lower middle income','Upper middle income', 'High income'])]
df_update = df_update.loc[:, ['country', 'year', 'Military expenditure (% of general government expenditure)']]

plt.figure(figsize=(20, 10))
def plotter_3(income_level):
    plt.title('Military expenditure (% of general government expenditure) for countries with different income levels', fontsize=17)
    plt.xlabel('Year', fontsize=17)
    plt.ylabel('Military expenditure (% of general government expenditure)', fontsize=17)
    plt.grid()
    years = [i for i in range(df['year'].min(), df['year'].max() + 1)]
    sns.lineplot(years, df_update[df_update['country'] == income_level]['Military expenditure (% of general government expenditure)'],
                 label=f'{income_level} level')
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.legend(fontsize=17)

    
plt.grid()
    
plotter_3('High income')
plotter_3('Upper middle income')
plotter_3('Lower middle income')
plotter_3('Low income')

