# Exno:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Coding and Output
```
import pandas as pd
data=pd.read_csv("SAMPLEIDS.csv")
data
```
<img width="1046" height="646" alt="Screenshot 2026-04-29 154600" src="https://github.com/user-attachments/assets/f05e3ea0-bfe1-4fd6-af1c-6dd036d04d3c" />

```
data.head()

```
<img width="972" height="224" alt="Screenshot 2026-04-29 155337" src="https://github.com/user-attachments/assets/c6d43e13-7063-4b7e-9cb0-622bf9128d37" />


```
data.tail()
```
<img width="1021" height="211" alt="Screenshot 2026-04-29 155535" src="https://github.com/user-attachments/assets/0db06c00-5266-43cf-b2b9-6aa85876eaba" />


```
data.isnull()
```
<img width="1037" height="575" alt="Screenshot 2026-04-29 155645" src="https://github.com/user-attachments/assets/5f31c89a-f6a6-490a-860e-ec9c0403d4c5" />


```
data.isnull().sum()
```
<img width="1029" height="255" alt="Screenshot 2026-04-29 155919" src="https://github.com/user-attachments/assets/f24ed8a2-52e5-402c-8aa5-d93e55027c35" />


```
data.isnull().any()
```
<img width="1042" height="270" alt="Screenshot 2026-04-29 160150" src="https://github.com/user-attachments/assets/95c0775e-0f1a-48aa-abce-c17d6bc5325c" />


```
data.dropna()
```
<img width="1032" height="394" alt="Screenshot 2026-04-29 160250" src="https://github.com/user-attachments/assets/bb998960-6ea6-41d5-949e-fe9e33718646" />


```
data.fillna(0)
```

<img width="1036" height="592" alt="Screenshot 2026-04-29 160335" src="https://github.com/user-attachments/assets/04651dd5-62f9-4755-8533-7add62fffbc0" />

```
data.fillna(method='ffill')
```
<img width="1038" height="582" alt="Screenshot 2026-04-29 160432" src="https://github.com/user-attachments/assets/9db3e67c-0fca-4173-8dcc-b23cacbca371" />


```
data.fillna(method='bfill')
```
<img width="1041" height="583" alt="Screenshot 2026-04-29 160840" src="https://github.com/user-attachments/assets/35fe4da6-3f42-4f34-9f3f-3e3c3c557dc2" />


```
data.fillna({'GENDER':'MALE','NAME':'SRI','ADDRESS':'POONAMALEE','M1':98,'M2':87,'M3':76,'M4':92,'TOTAL':305,'AVG':89.999999})
```

<img width="1049" height="580" alt="Screenshot 2026-04-29 161059" src="https://github.com/user-attachments/assets/07fb53b5-4894-48db-a0a8-3f2bb562ee15" />


IQR(Inter Quartile Range)
```
import pandas as pd
ir=pd.read_csv("iris.csv")
ir
```
<img width="1042" height="427" alt="Screenshot 2026-04-29 161201" src="https://github.com/user-attachments/assets/fb0b2d70-405a-4098-b8a3-764038300bf2" />


```
ir.describe()
```
<img width="1031" height="291" alt="Screenshot 2026-04-29 161435" src="https://github.com/user-attachments/assets/673ce2c3-e024-43ce-84fa-0caa653bbedf" />



```
ir.shape
```
<img width="968" height="60" alt="Screenshot 2026-04-29 161534" src="https://github.com/user-attachments/assets/19769f3b-d136-4f3e-ba60-12b544ee6438" />


```
ir.info()
```
<img width="1045" height="255" alt="Screenshot 2026-04-29 161630" src="https://github.com/user-attachments/assets/1a7a700b-7116-4b27-a3be-f17d9878923d" />


```
import seaborn as sns
sns.boxplot(x='sepal_width',data=ir)
```
<img width="1008" height="504" alt="Screenshot 2026-04-29 161812" src="https://github.com/user-attachments/assets/c20aa9c5-3e5e-4335-8a08-80611cc3c8c8" />


```
 q1=ir.sepal_width.quantile(0.25)
 q3=ir.sepal_width.quantile(0.75)
 iqr=q3-q1
 print(iqr)
```
<img width="367" height="119" alt="Screenshot 2026-04-29 161854" src="https://github.com/user-attachments/assets/e45c228e-4c82-494e-b278-cf152c1092b8" />



```
 out=ir[((ir.sepal_width<(q1-1.5*iqr))|(ir.sepal_width>(q3+1.5*iqr)))]
 out['sepal_width']
```

<img width="876" height="180" alt="Screenshot 2026-04-29 161939" src="https://github.com/user-attachments/assets/b70aeb1a-2268-48e7-8aff-b181916114eb" />


```
 nor=ir[~((ir.sepal_width<(q1-1.5*iqr))|(ir.sepal_width>(q3+1.5*iqr)))]
 nor['sepal_width']
```
<img width="905" height="276" alt="Screenshot 2026-04-29 162017" src="https://github.com/user-attachments/assets/00f7ab78-c109-4922-801b-6e434f0e7a7c" />



```
sns.boxplot(x='sepal_width',data=nor)
```
<img width="958" height="489" alt="Screenshot 2026-04-29 162239" src="https://github.com/user-attachments/assets/91bdd927-500f-43b4-b6b2-72225a34a8c5" />



### Z-SCORE
```
import numpy as np
import pandas as pd
df=pd.read_csv("heights.csv")
df
```
<img width="417" height="464" alt="Screenshot 2026-04-29 181814" src="https://github.com/user-attachments/assets/66de9920-82c0-4eb2-a9a0-8cb52fa71fcd" />

```
import scipy.stats as stats
q1 = df['height'].quantile(0.25)
q2 = df['height'].quantile(0.5)
q3 = df['height'].quantile(0.75)
iqr = q3-q1
iqr
```
<img width="447" height="177" alt="Screenshot 2026-04-29 181945" src="https://github.com/user-attachments/assets/897be8fd-8bbc-429c-b54b-b2f2643c9b93" />

```
low = q1 - 1.5*iqr
print(low)
high = q3 + 1.5*iqr
print(high)
```
<img width="417" height="464" alt="Screenshot 2026-04-29 181814" src="https://github.com/user-attachments/assets/eafc95ce-f5d1-4564-b428-2acc9065b8fd" />

```
df1 = df[((df['height'] >=low)& (df['height'] <=high))]
df1
```
<img width="544" height="379" alt="Screenshot 2026-04-29 182146" src="https://github.com/user-attachments/assets/4a19ff4a-e3ad-46dc-a683-da22a6ae3472" />

```
z = np.abs(stats.zscore(df['height']))
z
```

<img width="501" height="308" alt="Screenshot 2026-04-29 182308" src="https://github.com/user-attachments/assets/b0ea53a2-c0b0-4201-a2fd-ccbd90edb476" />


```
df1 = df[z<3]
df1
```
<img width="277" height="404" alt="Screenshot 2026-04-29 182359" src="https://github.com/user-attachments/assets/22130848-75af-471f-85db-9e5eed14dc8d" />



# Result
Thus we have cleaned the data and removed the outliers by detection using IQR and Z-score method.
