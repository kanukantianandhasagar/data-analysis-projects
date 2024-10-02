#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


df = pd.read_csv(r"C:\Users\k.anandhasagar\Desktop\data.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.columns=df.columns.str.strip()


# In[6]:


df.columns=df.columns.str.lower()


# In[7]:


df.info()


# ### Data cleaning

# In[8]:


df.drop(columns = "unnamed: 0",inplace=True)


# In[9]:


df.duplicated().sum()


# In[10]:


df.head()


# In[11]:


print("Shape of DataFrame:", df.shape)


# In[12]:


print("Description of DataFrame:")
round(df.describe())


# In[13]:


### To find the anomolities in data
for i in df.columns:
    print('*'*20,i,'*'*20)
    print(df[i].unique())


# In[14]:


import datetime as dt
df["doj"]=pd.to_datetime(df["doj"]).dt.date
df["dol"].replace("present",dt.datetime.today(),inplace=True)
df['dol'] = pd.to_datetime(df['dol']).dt.date
## We will engineer this feature from DOJ and DOL as we are only concerned with how many years the person has worked
## in the organisation.
df['Period'] = pd.to_datetime(df["dol"]).dt.year - pd.to_datetime(df['doj']).dt.year

##We only need DOB year,so we will convert DOB column from timestamp to year
df['dob'] = pd.to_datetime(df['dob']).dt.year
df.head(5)


# In[15]:


##We also know graduation year contains 0 value,we need to impute it with mode before engineering new feature from this.
## we are using dataset.GraduationYear.mode()[0] as it return a series unlike df.mean/mode
df['graduationyear'].replace(0,df.graduationyear.mode()[0],inplace=True)
df['graduationyear']=pd.to_datetime(df['graduationyear'])
df['gyear']=df['graduationyear'].dt.year

### New columns which can used to the know 
df['12GradAge']=abs(df['12graduation']-df['dob'])
df['GradAge']=abs(df['gyear']-df['dob'])


# In[16]:


# no of 0's per column
(df==0).astype(int).sum(axis=0)


# In[17]:


df.isin([-1, 'NaN']).sum()


# In[18]:


### Here we could have compared modes of all the columns and then could have selected the mode out of the resulting modes
### But from intuition,i thought mostly people from particular specialization choose desired designations.
df[df["designation"]=="get"][['designation','jobcity','salary','specialization']]


# In[19]:


#for people with mechanical engineering,it gives the mode value which will be replaced with the 'get'.
mech = df[df['specialization'].isin(['mechanical engineering','mechanical and automation'])]['designation'].mode()[0]
#for people with electronics and electrical engineering,it gives the mode value which will be replaced with the 'get'.
eee = df[df['specialization']==('electronics and electrical engineering')]['designation'].mode()[0]
print(f'mode for mechanical:  {mech}\nmode for EEE:  {eee}')


# In[20]:


#For mechanical domain
df.loc[df['specialization'].isin(['mechanical engineering','mechanical and automation']),'designation'].replace('get',mech,inplace=True)
#for EEE domain,as all previous get's will be replaced,we can replace the remaining directly without conditions
df['designation'].replace('get',eee,inplace=True)


# In[21]:


### we do not want our data to be case sensitive in jobcity
### ,because it will effect our analysis.so let us replace -1 with some string and then apply title method to it.
df['jobcity'].replace("-1",'unknown',inplace=True)
df['jobcity'].apply(lambda x:x.title())


# In[22]:


df[df["jobcity"]=='unknown']


# In[23]:


df[df["jobcity"]=="unknown"][["designation","12GradAge","GradAge","jobcity","gender","10percentage","10board","12percentage","12board","degree","collegestate","specialization"]].mode()


# In[24]:


### cleaning the column which have similar meaning but has spelling difference orelse it will effect the distribution.
df["jobcity"].replace("Bangalore","Bengaluru",inplace=True)
df["jobcity"].replace("Banaglore","Bengaluru",inplace=True)
df["jobcity"].replace("Chennai, Bangalore","Bengaluru",inplace=True)
df["jobcity"].replace(" Bangalore","Bengaluru",inplace=True)
df["jobcity"].replace("Bangalore ","Bengaluru",inplace=True)
df["jobcity"].replace("Banglore","Bengaluru",inplace=True)
df["jobcity"].replace("Jaipur ","Jaipur",inplace=True)
df["jobcity"].replace("Gandhinagar","Gandhi Nagar",inplace=True)
df["jobcity"].replace("Bangalore ","Bengaluru",inplace=True)
df["jobcity"].replace("Jaipur ","Jaipur",inplace=True)
df["jobcity"].replace("Gandhinagar","Gandhi Nagar",inplace=True)
df["jobcity"].replace("Hyderabad ","Hyderabad",inplace=True)
df["jobcity"].replace("Hyderabad(Bhadurpally)","Hyderabad",inplace=True)
df["jobcity"].replace("Bhubaneswar ","Bhubaneswar",inplace=True)
df["jobcity"].replace("Delhi/Ncr","Delhi",inplace=True)
df["jobcity"].replace("Nagpur ","Nagpur",inplace=True)
df["jobcity"].replace("Pune ","Pune",inplace=True)
df["jobcity"].replace("Trivandrum ","Trivandrum",inplace=True)
df["jobcity"].replace("Thiruvananthapuram","Trivandrum",inplace=True)


# In[25]:


### First,we saw the frequent(mode) values in other columns when we have a missing value in our target column('Jobcity')
### Now,we will find list of modes of other columns when they have the above found frequent value in their respective column.
### In this way,we are able to include the presence of all columns in predicting our best shot for the missing value.

best_mode = []
best_mode.append(df[df["designation"]=="software engineer"]["jobcity"].mode().to_list()[0])
best_mode.append(df[df["gender"]=="m"]["jobcity"].mode().to_list()[0])
best_mode.append(df[df["10percentage"]==76]["jobcity"].mode().to_list()[0])
best_mode.append(df[df["10board"]=="cbse"]["jobcity"].mode().to_list()[0])
best_mode.append(df[df["12percentage"]==64]["jobcity"].mode().to_list()[0])
best_mode.append(df[df["12board"]=="cbse"]["jobcity"].mode().to_list()[0])
best_mode.append(df[df["collegegpa"]==70]["jobcity"].mode().to_list()[0])
best_mode.append(df[df["salary"]==200000]["jobcity"].mode().to_list()[0])
best_mode.append(df[df["degree"].str.startswith("B.Tech/")]["jobcity"].mode().to_list()[0])
best_mode.append(df[df["specialization"].str.startswith("electronics and communication eng")]["jobcity"].mode().to_list()[0])
best_mode.append(df[df["collegestate"].str.startswith("Uttar Pradesh")]["jobcity"].mode().to_list()[0])
best_mode


# In[26]:


### We can see mode from the best_mode list is 'Bangalore'
df["jobcity"].replace("unknown",'Bengaluru',inplace=True)


# In[27]:


df[df["10board"]=="0"][["designation","12GradAge","GradAge","jobcity","gender","10percentage","10board","12percentage","12board","degree","specialization","collegestate"]].mode()


# In[28]:


df[df["12board"]=="0"][["designation","12GradAge","GradAge","jobcity","gender","10percentage","10board","12percentage","12board","degree","specialization","collegestate"]].mode()


# In[29]:


### Same process as above written for jobcity
best_value2=[]
best_value2.append(df[df["designation"]=="software engineer"]["10board"].mode().to_list()[0])
best_value2.append(df[df["gender"]=="m"]["10board"].mode().to_list()[0])
best_value2.append(df[df["10percentage"]==75]["10board"].mode().to_list()[0])
best_value2.append(df[df["jobcity"]=="Bengaluru"]["10board"].mode().to_list()[0])
best_value2.append(df[df["12percentage"]==65]["10board"].mode().to_list()[0])
best_value2.append(df[df["collegegpa"]==65]["10board"].mode().to_list()[0])
best_value2.append(df[df["salary"]==400000]["10board"].mode().to_list()[0])
best_value2.append(df[df["degree"].str.startswith("B.Tech/")]["10board"].mode().to_list()[0])
best_value2.append(df[df["specialization"].str.startswith("computer eng")]["10board"].mode().to_list()[0])
best_value2.append(df[df["collegestate"].str.startswith("Tamil Nadu")]["10board"].mode().to_list()[0])
best_value2


# In[30]:


### Replacing with the mode of the best_value list(visually as it is a small list orelse could have written code for it.)
df['10board'].replace("0",'cbse',inplace=True)


# In[31]:


### From what i found from above,we can be sure that 12 board missing value can be replaced with 'cbse' 
### as most of the people do 12th also from the same board.(general observation,can also be proved)
df['12board'].replace("0",'cbse',inplace=True)


# In[32]:


sns.boxplot(df['domain'])
plt.show()


# In[33]:


## As we can see outlier,it is better to use median to replace the missing values.
df['domain'].replace(-1,df['domain'].median(),inplace=True)
df.head()


# In[34]:


#replacing the redundant values of the 12board column with 'state','cbse','icse' and 'n/a'
replace_list_state=['board of intermediate education,ap', 'state board',
       'mp board',  'karnataka pre university board', 'up',
       'p u board, karnataka', 'dept of pre-university education', 'bie',
       'kerala state hse board', 'up board', 'bseb', 'chse', 'puc',
       ' upboard',
       'state  board of intermediate education, andhra pradesh',
       'karnataka state board',
       'west bengal state council of technical education', 'wbchse',
       'maharashtra state board', 'ssc',
       'sda matric higher secondary school', 'uttar pradesh board', 'ibe',
       'chsc', 'board of intermediate', 'upboard', 'sbtet',
       'hisher seconadry examination(state board)', 'pre university',
       'borad of intermediate', 'j & k board',
       'intermediate board of andhra pardesh', 'rbse',
       'central board of secondary education', 'jkbose', 'hbse',
       'board of intermediate education', 'state', 'ms board', 'pue',
       'intermediate state board', 'stateboard', 'hsc',
       'electonincs and communication(dote)', 'karnataka pu board',
       'government polytechnic mumbai , mumbai board', 'pu board',
       'baord of intermediate education', 'apbie', 'andhra board',
       'tamilnadu stateboard',
       'west bengal council of higher secondary education',
       'cbse,new delhi', 'u p board', 'intermediate', 'biec,patna',
       'diploma in engg (e &tc) tilak maharashtra vidayapeeth',
       'hsc pune', 'pu board karnataka', 'kerala', 'gsheb',
       'up(allahabad)', 'nagpur', 'st joseph hr sec school',
       'pre university board', 'ipe', 'maharashtra', 'kea', 'apsb',
       'himachal pradesh board of school education', 'staae board',
       'international baccalaureate (ib) diploma', 'nios',
       'karnataka board of university',
       'board of secondary education rajasthan', 'uttarakhand board',
       'ua', 'scte vt orissa', 'matriculation',
       'department of pre-university education', 'wbscte',
       'preuniversity board(karnataka)', 'jharkhand accademic council',
       'bieap', 'msbte (diploma in computer technology)',
       'jharkhand acamedic council (ranchi)',
       'department of pre-university eduction', 'biec',
       'sjrcw', ' board of intermediate', 'msbte',
       'sri sankara vidyalaya', 'chse, odisha', 'bihar board',
       'maharashtra state(latur board)', 'rajasthan board', 'mpboard',
       'state board of technical eduction panchkula', 'upbhsie', 'apbsc',
       'state board of technical education and training',
       'secondary board of rajasthan',
       'tamilnadu higher secondary education board',
       'jharkhand academic council',
       'board of intermediate education,hyderabad', 'up baord', 'pu',
       'dte', 'board of secondary education', 'pre-university',
       'board of intermediate education,andhra pradesh',
       'up board , allahabad', 'srv girls higher sec school,rasipuram',
       'intermediate board of education,andhra pradesh',
       'intermediate board examination',
       'department of pre-university education, bangalore',
       'stmiras college for girls', 'mbose',
       'department of pre-university education(government of karnataka)',
       'dpue', 'msbte pune', 'board of school education harayana',
       'sbte, jharkhand', 'bihar intermediate education council, patna',
       'higher secondary', 's j polytechnic', 'latur',
       'board of secondary education, rajasthan', 'jyoti nivas', 'pseb',
       'biec-patna', 'board of intermediate education,andra pradesh',
       'chse,orissa', 'pre-university board', 'mp', 'intermediate board',
       'govt of karnataka department of pre-university education',
       'karnataka education board',
       'board of secondary school of education', 'pu board ,karnataka',
       'karnataka secondary education board', 'karnataka sslc',
       'board of intermediate ap', 'u p', 'state board of karnataka',
       'directorate of technical education,banglore', 'matric board',
       'andhpradesh board of intermediate education',
       'stjoseph of cluny matrhrsecschool,neyveli,cuddalore district',
       'bte up', 'scte and vt ,orissa', 'hbsc',
       'jawahar higher secondary school', 'nagpur board', 'bsemp',
       'board of intermediate education, andhra pradesh',
       'board of higher secondary orissa',
       'board of secondary education,rajasthan(rbse)',
       'board of intermediate education:ap,hyderabad', 'science college',
       'karnatak pu board', 'aissce', 'pre university board of karnataka',
       'bihar', 'kerala state board', 'uo board', 
       'karnataka board', 'tn state board',
       'kolhapur divisional board, maharashtra',
       'jaycee matriculation school',
       'board of higher secondary examination, kerala',
       'uttaranchal state board', 'intermidiate', 'bciec,patna', 'bice',
       'karnataka state', 'state broad', 'wbbhse', 'gseb',
       'uttar pradesh', 'ghseb', 'board of school education uttarakhand',
       'gseb/technical education board', 'msbshse,pune',
       'tamilnadu state board', 'board of technical education',
       'kerala university', 'uttaranchal shiksha avam pariksha parishad',
       'chse(concil of higher secondary education)',
       'bright way college, (up board)', 'board of intermidiate',
       'higher secondary state certificate', 'karanataka secondary board',
       'maharashtra board', 'cgbse', 'diploma in computers', 'bte,delhi',
       'rajasthan board ajmer', 'mpbse', 'pune board',
       'state board of technical education', 'gshseb',
       'amravati divisional board', 'dote (diploma - computer engg)',
       'karnataka pre-university board', 'jharkhand board',
       'punjab state board of technical education & industrial training',
       'department of technical education',
       'sri chaitanya junior kalasala', 'state board (jac, ranchi)',
       'aligarh muslim university', 'tamil nadu state board', 'hse',
       'karnataka secondary education', 'state board ',
       'karnataka pre unversity board',
       'ks rangasamy institute of technology',
       'karnataka board secondary education', 'narayana junior college',
       'bteup', 'board of intermediate(bie)', 'hsc maharashtra board',
       'tamil nadu state', 'uttrakhand board', 'psbte',
       'stateboard/tamil nadu', 'intermediate council patna',
       'technical board, punchkula', 'board of intermidiate examination',
       'sri kannika parameswari highier secondary school, udumalpet',
       'ap board', 'nashik board', 'himachal pradesh board',
       'maharashtra satate board',
       'andhra pradesh board of secondary education',
       'tamil nadu polytechnic',
       'maharashtra state board mumbai divisional board',
       'department of pre university education',
       'dav public school,hehal', 'board of intermediate education, ap',
       'rajasthan board of secondary education',
       'department of technical education, bangalore', 'chse,odisha',
       'maharashtra nasik board',
       'west bengal council of higher secondary examination (wbchse)',
       'holy cross matriculation hr sec school', 'cbsc',
       'pu  board karnataka', 'biec patna', 'kolhapur', 'bseb, patna',
       'up board allahabad', 'nagpur board,nagpur', 'diploma(msbte)',
       'dav public school', 'pre university board, karnataka',
       'ssm srsecschool', 'state bord', 'jstb,jharkhand',
       'intermediate board of education', 'mp board bhopal', 'pub',
       'madhya pradesh board', 'bihar intermediate education council',
       'west bengal council of higher secondary eucation',
        'mpc',
       'certificate for higher secondary education (chse)orissa',
       'maharashtra state board for hsc',
       'board of intermeadiate education', 'latur board',
       'andhra pradesh', 'karnataka pre-university',
       'lucknow public college', 'nagpur divisional board',
       'ap intermediate board', 'cgbse raipur', 'uttranchal board',
       'jiec', 
       'bihar school examination board patna',
       'state board of technical education harayana', 'mp-bse',
       'up bourd', 'dav public school sec 14',
       'haryana state board of technical education chandigarh',
       'council for indian school certificate examination',
       'jaswant modern school', 'madhya pradesh open school',
       'aurangabad board', 'j&k state board of school education',
       'diploma ( maharashtra state board of technical education)',
       'board of technicaleducation ,delhi',
       'maharashtra state boar of secondary and higher secondary education',
       'hslc (tamil nadu state board)',
       'karnataka state examination board', 'puboard', 'nasik',
       'west bengal board of higher secondary education',
       'up board,allahabad', 'board of intrmediate education,ap', 
       'karnataka state pre- university board',
       'state board - west bengal council of higher secondary education : wbchse',
       'maharashtra state board of secondary & higher secondary education',
       'biec, patna', 'state syllabus', 'cbse board', 'scte&vt',
       'board of intermediate,ap',
       'secnior secondary education board of rajasthan',
       'maharashtra board, pune', 'rbse (state board)',
       'board of intermidiate education,ap',
       'board of high school and intermediate education uttarpradesh',
       'higher secondary education',
       'board fo intermediate education, ap', 'intermedite',
       'ap board for intermediate education', 'ahsec',
       'punjab state board of technical education & industrial training, chandigarh',
       'state board - tamilnadu', 'jharkhand acedemic council',
       'scte & vt (diploma)', 'karnataka pu',
       'board of intmediate education ap', 'up-board',
       'boardofintermediate','intermideate','up bord','andhra pradesh state board','gujarat board']


# In[35]:


#replacing the redundant values of the 12board column with 'state','cbse','icse' 
for i in replace_list_state:
    df['12board'].replace(i,'state',inplace=True)

replace_list_cbse=['cbse', 
       'all india board', 
       'central board of secondary education, new delhi', 'cbese']
for i in replace_list_cbse:
    df['12board'].replace(i,'cbse',inplace=True)

replace_list_icse=[ 'isc', 'icse', 'isc board', 'isce', 'cicse',
       'isc board , new delhi']
for i in replace_list_icse:
    df['12board'].replace(i,'icse',inplace=True)

df['12board'].unique()


# In[36]:


df['12board'].value_counts()


# In[37]:


specialization_map = \
{'electronics and communication engineering' : 'EC',
 'computer science & engineering' : 'CS',
 'information technology' : 'CS' ,
 'computer engineering' : 'CS',
 'computer application' : 'CS',
 'mechanical engineering' : 'ME',
 'electronics and electrical engineering' : 'EC',
 'electronics & telecommunications' : 'EC',
 'electrical engineering' : 'EL',
 'electronics & instrumentation eng' : 'EC',
 'civil engineering' : 'CE',
 'electronics and instrumentation engineering' : 'EC',
 'information science engineering' : 'CS',
 'instrumentation and control engineering' : 'EC',
 'electronics engineering' : 'EC',
 'biotechnology' : 'other',
 'other' : 'other',
 'industrial & production engineering' : 'other',
 'chemical engineering' : 'other',
 'applied electronics and instrumentation' : 'EC',
 'computer science and technology' : 'CS',
 'telecommunication engineering' : 'EC',
 'mechanical and automation' : 'ME',
 'automobile/automotive engineering' : 'ME',
 'instrumentation engineering' : 'EC',
 'mechatronics' : 'ME',
 'electronics and computer engineering' : 'CS',
 'aeronautical engineering' : 'ME',
 'computer science' : 'CS',
 'metallurgical engineering' : 'other',
 'biomedical engineering' : 'other',
 'industrial engineering' : 'other',
 'information & communication technology' : 'EC',
 'electrical and power engineering' : 'EL',
 'industrial & management engineering' : 'other',
 'computer networking' : 'CS',
 'embedded systems technology' : 'EC',
 'power systems and automation' : 'EL',
 'computer and communication engineering' : 'CS',
 'information science' : 'CS',
 'internal combustion engine' : 'ME',
 'ceramic engineering' : 'other',
 'mechanical & production engineering' : 'ME',
 'control and instrumentation engineering' : 'EC',
 'polymer technology' : 'other',
 'electronics' : 'EC'}


# In[38]:


df['specialization'] = df['specialization'].map(specialization_map)
df['specialization'].unique()


# In[39]:


df.drop(columns=['collegeid','collegecityid','collegecitytier'],axis=1,inplace=True)


# In[40]:


df.columns


# In[41]:


### Salary less than 50000 people might have entered their montly income rather than yearly
df.loc[df['salary']<=50000,'salary']*=12
lst = ['computerprogramming','electronicsandsemicon','computerscience','mechanicalengg','electricalengg','telecomengg','civilengg']
for i in lst:
    df[i].replace(-1,0,inplace=True)


# ### data visualization

# ## univariate analysis

# Univariate Analysis -> PDF, Histograms, Boxplots, Countplots, etc.. Find the outliers in each numerical column Understand the probability and frequency distribution of each numerical column Understand the frequency distribution of each categorical Variable/Column Mention observations after each plot.

# In[42]:


plt.figure(figsize=(15,5))
colors = sns.color_palette('bright',n_colors=2)
sns.FacetGrid(df, col="gender", palette=colors) \
   .map(sns.distplot, "salary",bins=50) \
   .add_legend()
plt.show()


# In[43]:


sns.countplot(x=df['gender'],palette='Set2')
print(df['gender'].value_counts())


# In[44]:


plt.figure(figsize=(10,5))
sns.boxplot(x='salary',y='gender',data=df)


# In[45]:


plt.figure(figsize=(15,5))
sns.boxplot(x='salary',y='specialization',data=df)
plt.suptitle('Salary levels by specialization')


# In[46]:


### Designation
popular_Designation = df['designation'].value_counts()[:20].index.tolist()
print(popular_Designation)


# In[47]:


### We want on
top_Designations = df[df['designation'].isin(popular_Designation)]
print(f"Unique professions : {len(df['designation'].unique())}")
top_Designations.head()


# In[48]:


plt.figure(figsize=(20,10))
sns.countplot(x='designation',hue='gender',data=top_Designations)
plt.xticks(fontsize=30,rotation=90)
plt.yticks(fontsize=30)
plt.show()


# In[49]:


custom_palette = ["#FF5733", "#33FF57", "#3357FF"]
sns.countplot(x=df['specialization'],palette=custom_palette)


# In[52]:


plt.figure(figsize=(20,10))
sns.barplot(x='designation',y='salary',hue='gender',data=top_Designations)
plt.xticks(fontsize=30,rotation=90)
plt.yticks(fontsize=30)
plt.show()


# In[53]:


### Now lets us see the high paying designations and their relation with respect to gender
high = list(df.sort_values("salary",ascending=False)["designation"].unique())[:20]
high_pay = df[df['designation'].isin(high)]
high_pay.head()


# In[54]:


plt.figure(figsize=(20,10))
sns.barplot(x='designation',y='salary',hue='gender',data=high_pay)
plt.xticks(fontsize=30,rotation=90)
plt.yticks(fontsize=30)
plt.show()


# In[55]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(numeric_only=True),annot=True)


# In[56]:


### Lets us check experience distribution of both male and female
plt.figure(figsize=(20,5))

sns.FacetGrid(high_pay, hue="gender").map(sns.distplot, "Period").add_legend()
print('For Whole dataset')
print(high_pay.groupby('gender').Period.mean())
print('*'*20)
print('For High_paying jobs')
print(df.groupby('gender').Period.mean())


# In[57]:


plt.figure(figsize=(20,10))
sns.boxplot(data=high_pay,x='Period',y='salary',hue='gender')


# In[58]:


plt.figure(figsize=(20,5))
sns.boxplot(data=high_pay,x='designation',y='Period',hue='gender')
plt.xticks(fontsize=20,rotation=90)
plt.show()


# In[59]:


sns.FacetGrid(data=high_pay,hue='gender') \
    .map(sns.scatterplot,'Period','salary') \
    .add_legend()


# In[60]:


### What is average experience of software engineer and software developer?
df[df.designation.isin(['software engineer','software developer']) & df.gender=='m']['Period'].mean()


# In[61]:


### Now let us check relation with collegegpa
### first check the distribution of gpa 
sns.FacetGrid(data=high_pay,col='gender').map(sns.distplot,'collegegpa').add_legend()


# In[62]:


sns.FacetGrid(data=high_pay,hue='gender') \
    .map(sns.scatterplot,'collegegpa','salary') \
    .add_legend()


# In[63]:


sns.barplot(data=df,x='specialization',y='salary',palette='Set2')


# In[64]:


# For the total Dataset
## Checking whether specialization has any effect on salary
plt.figure(figsize=(20,10))
palette = [(0.8666666666666667, 0.5176470588235295, 0.3215686274509804),(0.2980392156862745, 0.4470588235294118, 0.6901960784313725)]
sns.barplot(data=df,x='specialization',y='salary',hue='gender',palette=palette)


# In[65]:


# for the dataset containing Highpaying Jobs
plt.figure(figsize=(20,10))
sns.barplot(data=high_pay,x='specialization',y='salary',hue='gender')


# In[66]:


### Lets us check salary with the College Tier
plt.figure(figsize=(10,5))
sns.barplot(data=high_pay,x='collegetier',y='salary',hue='gender')


# In[67]:


high_pay.groupby('collegetier').gender.value_counts()


# In[68]:


plt.figure(figsize=(15,5))
df['AverageScore']=(df['logical']+df['quant']+df['english'])/3
df['Acadperf']=df['10percentage']+df['12percentage']+df['collegegpa']/3
plt.subplot(1,2,1)
sns.regplot(x='AverageScore',y='salary',data=df)
plt.subplot(1,2,2)
sns.regplot(x='Acadperf',y='salary',data=df)
plt.show()


# In[69]:


plt.figure(figsize=(55,15))
sns.countplot(x="jobcity",data=high_pay,hue="gender")
plt.xticks(fontsize=38,rotation=90)
plt.yticks(fontsize=38)


# In[70]:


plt.figure(figsize=(55,15))
sns.countplot(data=df,x='collegestate',palette='Set2')
plt.xticks(fontsize=38,rotation=90)
plt.show()


# In[71]:


custom_palette = ["#FF5733", "#33FF57", "#3357FF"]
plt.figure(figsize=(55,15))
sns.countplot(data=df,x='collegestate',palette=custom_palette)
plt.xticks(fontsize=38,rotation=90)
plt.show()


# In[ ]:




