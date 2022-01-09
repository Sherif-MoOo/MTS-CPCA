#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''import numpy  as np
import pandas as pd
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing'''
import h5py


# In[2]:


with h5py.File('data.h5', 'r') as hf:
    X = hf['EEG_values'][:]             #Samples tensor
    y = hf['target_values'][:]          #Targets matrix


# In[3]:


X.shape


# In[4]:


'''No_Activity_Events = np.where(y==0)[0]
len(No_Activity_Events)'''


# In[5]:


'''No_Activity_Events = np.where(y==0)[0]
len(No_Activity_Events)
Events = pd.Series(y).to_frame().rename(columns={0: "Events"})['Events'].astype(str)
type_counts = Events.value_counts()
Coutns = type_counts.sum(axis = 0)
for i in range(len(type_counts)):
    type_counts.iloc[i] = int((type_counts.iloc[i]/Coutns)*100)
ax =type_counts.plot(kind='bar' , figsize =(10,10)  , fontsize = 12);
plt.suptitle('Data_Balance', fontsize=18);
plt.ylabel('Percentage', fontsize=18);
plt.xlabel('Events', fontsize=18);'''


# In[6]:


channels_order = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']


# In[7]:


'''list_=[]
for i in range (0, len(No_Activity_Events), 2):
    list_ = list_ + [No_Activity_Events[i]]
y = np.delete(y, list_, 0)
X = np.delete(X, list_, 0)
No_Activity_Events = np.where(y==0)[0]
list_=[]
for i in range (0, len(No_Activity_Events), 2):
    list_ = list_ + [No_Activity_Events[i]]
y = np.delete(y, list_, 0)
X = np.delete(X, list_, 0)
No_Activity_Events = np.where(y==0)[0]
len(No_Activity_Events)'''


# In[8]:


#Input tensor X as shape of M: Samples , ni: length of time sereis , m: Numper of features 


# In[9]:


class PCA_MTS():
    Index_Samples             = 0
    Index_length_time_sereis  = 1
    Index_features            = 2
    X_normalized = 0
    
    def __init__(self,X):
        
        print("Pls make sure that Input tensor X as shape of M: Samples , ni: length of time sereis , m: Numper of features ")        
        self.X = X
        
        def global_imports(modulename,shortname = None, asfunction = False):
            if shortname is None: 
                shortname = modulename
            if asfunction is False:
                globals()[shortname] = __import__(modulename)
            else:        
                globals()[shortname] = eval(modulename + "." + shortname) 
                
        global_imports("numpy","np")
        global_imports("tensorflow","tf")
        global_imports("tensorflow_probability","tfp")
        global_imports("seaborn","sns")
        global_imports("matplotlib","plt")

        
        #mean of each sample per m features cross ni length
        mean_vector_i = tf.divide(tf.reduce_sum(X, self.Index_length_time_sereis), X.shape[self.Index_length_time_sereis])
        #This result in tensor of shape (M , m)

        #Now broad-castting the tensor intp (M, ni ,m)
        mean_vector_i = np.tile(mean_vector_i, (1,X.shape[self.Index_length_time_sereis])).reshape(X.shape[self.Index_Samples],
                                                                                      X.shape[self.Index_length_time_sereis],
                                                                                      X.shape[self.Index_features])

        PCA_MTS.X_normalized  = tf.subtract(X,mean_vector_i).numpy() #Getting the normalized tensor

        

    def Stats_COV(self):
        
        State_x = PCA_MTS.X_normalized

        DeNormalized_Segma = tfp.stats.covariance(State_x, sample_axis=1, event_axis=2, keepdims=False, name=None)
    
        Segma_COV          = tf.divide(tf.reduce_sum(DeNormalized_Segma, PCA_MTS.Index_Samples), X.shape[PCA_MTS.Index_Samples])
    
        return Segma_COV 
    
    def Correlation(self , Columns , figs = (10,10) ,titles = 20):
        
        store = PCA_MTS.X_normalized
        
        std_x = PCA_MTS.X_normalized
 
    
        std     = tf.math.reduce_std(
            std_x, axis=1, keepdims=False, name=None)
    
        std     = np.tile(std, (1,X.shape[PCA_MTS.Index_length_time_sereis])).reshape(X.shape[PCA_MTS.Index_Samples],
                                                                                      X.shape[PCA_MTS.Index_length_time_sereis],
                                                                                      X.shape[PCA_MTS.Index_features])
        std_x   =  tf.divide(std_x,std)
    
        fig, ax = plt.pyplot.subplots(figsize= figs)
        
        PCA_MTS.X_normalized = std_x
    
        sns.heatmap(tf.math.abs(self.Stats_COV()),
                     cbar=True,
                     annot=True,
                     square=True,
                     fmt='.2f',
                     annot_kws={'size': 12},
                     cmap='coolwarm',                 
                     yticklabels = Columns,
                     xticklabels = Columns,
                     ax = ax)
        plt.pyplot.title('Covariance matrix showing abs correlation coefficients', size = titles)
        plt.pyplot.tight_layout()
        plt.pyplot.show()
        
        PCA_MTS.X_normalized = store
    
        return None


# In[10]:


a = PCA_MTS(X)


# ![image.png](attachment:image.png)

# In[11]:


a.Stats_COV()


# In[12]:


a.Correlation(channels_order)


# In[ ]:





# In[ ]:




