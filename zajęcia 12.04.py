
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris


# In[2]:


iris=load_iris()


# In[3]:


from sklearn.model_selection import train_test_split


# In[4]:


import pandas as pd


# In[5]:


import numpy as np


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)


# In[7]:


X_train.shape


# In[8]:


X_test.shape


# In[10]:


from sklearn.neighbors import KNeighborsClassifier


# In[11]:


ds=KNeighborsClassifier(3)


# In[12]:


ds.fit(X_train, y_train)


# In[13]:


y_pred=ds.predict(X_test)


# In[14]:


from sklearn.metrics import accuracy_score, classification_report


# In[15]:


print(accuracy_score(y_pred, y_test))


# In[16]:


print(classification_report(y_pred, y_test))


# In[17]:


import numpy as np


# In[18]:


total_rzuty = 30
liczba_orlow = 24
prawd_orla = 0.5


# In[19]:


experiment = np.random.randint(0,2,total_rzuty)


# In[20]:


print("Dane Eksperymentalne :{}".format(experiment))


# In[21]:


ile_orlow = experiment[experiment==1].shape[0]


# In[22]:


print("Liczba orłów w eksperymencie:", ile_orlow )


# In[25]:


def rzut_moneta_eksperyment(ile_razy_powtorzyc):
    head_count = np.empty([ile_razy_powtorzyc,1],dtype=int)
    for times in np.arange(ile_razy_powtorzyc):
        experiment = np.random.randint(0,2, total_rzuty)
        head_count[times] = experiment[experiment==1].shape[0]
    return head_count


# In[36]:


head_count = rzut_moneta_eksperyment(10000)
head_count[:10]
print('Wymiar:{} \n Typ: {}'.format(head_count.shape,type(head_count)))


# In[37]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(color_codes = True)


# In[38]:


sns.distplot(head_count, kde=False)
sns.distplot(head_count, kde=True)


# In[41]:


print('Otrzymaliśmy {} orłów. Co stanowiło {} procent'.format(head_count[head_count >= 24].shape[0],(head_count[head_count>=24].shape[0]/float(head_count.shape[0])*100)))


# In[45]:


def coin_toss_experiment(times_to_repeat):

    head_count = np.empty([times_to_repeat,1], dtype=int)
    experiment = np.random.randint(0,2,[times_to_repeat,total_rzuty])
    return experiment.sum(axis=1)


# In[47]:


coin_toss_experiment(10)

