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
<img width="1046" height="646" alt="Screenshot 2026-04-29 154600" src="https://github.com/user-attachments/assets/142bcd47-7061-470a-9a28-59265ead02ec" />
```
data.head()
```
<img width="972" height="224" alt="Screenshot 2026-04-29 155337" src="https://github.com/user-attachments/assets/da14a775-b7a2-40d6-a185-b6acf6ca6f08" />
```
data.tail()
```
<img width="1021" height="211" alt="Screenshot 2026-04-29 155535" src="https://github.com/user-attachments/assets/12e55b3e-0273-449e-847f-0f5bbe58e916" />
```
data.isnull()
```
<img width="1037" height="575" alt="Screenshot 2026-04-29 155645" src="https://github.com/user-attachments/assets/0d7edaab-e9c0-4e78-a7e2-5d3577bc9193" />
```
data.isnull().sum()
```
<img width="1029" height="255" alt="Screenshot 2026-04-29 155919" src="https://github.com/user-attachments/assets/49abbe44-b2b9-42b4-8b2c-6e4aaabaa5ec" />
```
data.isnull().any()
```
<img width="1042" height="270" alt="Screenshot 2026-04-29 160150" src="https://github.com/user-attachments/assets/ffcc8f30-e181-4a20-88a0-9189097651e9" />
```
data.dropna()
```
<img width="1032" height="394" alt="Screenshot 2026-04-29 160250" src="https://github.com/user-attachments/assets/4d2c9f6f-9114-4511-9dff-4f5e05d5f0be" />
```
data.fillna(0)
```
<img width="1036" height="592" alt="Screenshot 2026-04-29 160335" src="https://github.com/user-attachments/assets/11ff91a8-12b5-4ceb-bf15-ad1d423e4d37" />
```
data.fillna(method='ffill')
```
<img width="1038" height="582" alt="Screenshot 2026-04-29 160432" src="https://github.com/user-attachments/assets/7152e67b-51d9-400a-a148-866a37a45e06" />
```
data.fillna(method='bfill')
```
<img width="1041" height="583" alt="Screenshot 2026-04-29 160840" src="https://github.com/user-attachments/assets/e7f1eb45-4952-4f9a-85e7-65cb7c2ea7c4" />
```
data.fillna({'GENDER':'MALE','NAME':'SRI','ADDRESS':'POONAMALEE','M1':98,'M2':87,'M3':76,'M4':92,'TOTAL':305,'AVG':89.999999})
```
<img width="1049" height="580" alt="Screenshot 2026-04-29 161059" src="https://github.com/user-attachments/assets/0664f567-7ec6-4004-8807-35300075b015" />

IQR(Inter Quartile Range)

```
import pandas as pd
ir=pd.read_csv("iris.csv")
ir
```
<img width="1042" height="427" alt="Screenshot 2026-04-29 161201" src="https://github.com/user-attachments/assets/c44db6be-5bb4-4e7d-b4de-b7bed41bfd73" />
```
ir.describe()
```
<img width="1031" height="291" alt="Screenshot 2026-04-29 161435" src="https://github.com/user-attachments/assets/5646b419-2316-4dd6-912d-f420e9d68744" />
```
ir.shape
```
<img width="968" height="60" alt="Screenshot 2026-04-29 161534" src="https://github.com/user-attachments/assets/691bf43d-2155-4a87-82ae-9d2b196a14ce" />
```
ir.info()
```
<img width="1045" height="255" alt="Screenshot 2026-04-29 161630" src="https://github.com/user-attachments/assets/e58034ab-7630-4479-9956-37c0af52177b" />
```
import seaborn as sns
sns.boxplot(x='sepal_width',data=ir)
```
<img width="1008" height="504" alt="Screenshot 2026-04-29 161812" src="https://github.com/user-attachments/assets/3b910a0c-4636-4a6a-b6d9-dc36e0ff82bb" />
```
 q1=ir.sepal_width.quantile(0.25)
 q3=ir.sepal_width.quantile(0.75)
 iqr=q3-q1
 print(iqr)
```
<img width="367" height="119" alt="Screenshot 2026-04-29 161854" src="https://github.com/user-attachments/assets/841dfecb-6aba-4ada-804f-ae4808501936" />
```
out=ir[((ir.sepal_width<(q1-1.5*iqr))|(ir.sepal_width>(q3+1.5*iqr)))]
out['sepal_width']
```
<img width="876" height="180" alt="Screenshot 2026-04-29 161939" src="https://github.com/user-attachments/assets/609605e9-f87c-4209-a1a6-fb039c7611f5" />
```
nor=ir[~((ir.sepal_width<(q1-1.5*iqr))|(ir.sepal_width>(q3+1.5*iqr)))]
nor['sepal_width']
```
<img width="905" height="276" alt="Screenshot 2026-04-29 162017" src="https://github.com/user-attachments/assets/75653135-5217-43e2-b74c-583e0a0a3d22" />
```
sns.boxplot(x='sepal_width',data=nor)
```
<img width="958" height="489" alt="Screenshot 2026-04-29 162239" src="https://github.com/user-attachments/assets/181f6d57-2552-4cdc-813a-e3f8308b7c57" />

### Z-SCORE
```
import numpy as np
import pandas as pd
df=pd.read_csv("heights.csv")
df
```
<img width="417" height="464" alt="Screenshot 2026-04-29 181814" src="https://github.com/user-attachments/assets/4bec4cf6-d7b9-4240-854d-2ade8afd74cb" />
```
import scipy.stats as stats
q1 = df['height'].quantile(0.25)
q2 = df['height'].quantile(0.5)
q3 = df['height'].quantile(0.75)
iqr = q3-q1
iqr

```
<img width="447" height="177" alt="image" src="https://github.com/user-attachments/assets/f7ce4f9f-9a84-4b2e-82c4-d51ecfec9c0d" />
```
low = q1 - 1.5*iqr
print(low)
high = q3 + 1.5*iqr
print(high)
```
<img width="306" height="135" alt="image" src="https://github.com/user-attachments/assets/ce284a09-0c52-467c-a3f7-ecdc71ba275e" />
```
df1 = df[((df['height'] >=low)& (df['height'] <=high))]
df1
```
<img width="544" height="379" alt="image" src="https://github.com/user-attachments/assets/99091c3e-9ac2-4739-9feb-4ca4c885d052" />
```
z = np.abs(stats.zscore(df['height']))
z
```
<img width="501" height="308" alt="image" src="https://github.com/user-attachments/assets/806054f7-a5dc-46b9-aa0e-bd2e0c301df2" />
```
df1 = df[z<3]
df1
```
<img width="277" height="404" alt="image" src="https://github.com/user-attachments/assets/9012af21-a650-4150-bfcc-6ef484187281" />

# Result
Thus we have cleaned the data and removed the outliers by detection using IQR and Z-score method.
