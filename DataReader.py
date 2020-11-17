#!/usr/bin/env python
# coding: utf-8

# In[83]:


import pandas as pd

import numpy as np

from sklearn.impute import SimpleImputer

from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as s

import warnings


# In[84]:


warnings.filterwarnings("ignore")


# In[85]:


class custom_data_reader:
    
    
    def read_csv(self,file_name,label_column_name,missing_values_threshold,apply_dim_reduction,explained_variance_fraction):
        
        datatype_dict = {}
        
        data = pd.read_csv(file_name)
        
        label = np.array(data[label_column_name])
        
        data.drop([label_column_name],axis=1,inplace=True)
        
        data.dropna(axis=1,inplace=True)
        
        temp_array = np.array(data)
        
        very_high_var_columns = np.argwhere(np.var(temp_array,axis=0) > 1000)
        
        for col_index in very_high_var_columns:
            
            series_data = pd.Series(temp_array[:,col_index[0]])
            
            single_col_category_data = series_data.astype('category')
            
            single_col_binned_data = pd.qcut(single_col_category_data,q=10).cat.codes
            
            temp_array[:,col_index[0]] = np.array(single_col_binned_data)
            
        data = pd.DataFrame(data=temp_array,columns=data.columns)
            
        for col in data.columns:
            
            datatype_dict[col] = self.determine_data_type(data,col,missing_values_threshold)
            
        filled_data = self.fillna(data,datatype_dict)
        
        smote_obj = SMOTE(random_state=42)
        
        filled_data,label = smote_obj.fit_resample(filled_data,label)
        
        label = label.reshape(label.shape[0],1)
        
        if apply_dim_reduction == True:
        
            dim_reduced_data = self.apply_PCA(filled_data,explained_variance_fraction)
            
            fig,axes = plt.subplots(1,dim_reduced_data.shape[1],figsize=(15,3))
            
            for i in range(dim_reduced_data.shape[1]):
                
                axes[i].hist(dim_reduced_data[:,i])
            
            dist_params_list = self.determine_distribution_type(dim_reduced_data)
            
            return [np.concatenate(tuple([dim_reduced_data,label]),axis=1),dist_params_list]
        
        else:
            
            fig,axes = plt.subplots(1,filled_data.shape[1],figsize=(15,3))
            
            for i in range(filled_data.shape[1]):
                
                axes[i].hist(filled_data[:,i])
            
            dist_params_list = self.determine_distribution_type(filled_data)
            
            return [np.concatenate(tuple([filled_data,label]),axis=1),dist_params_list]
            
            
        
    def determine_data_type(self,data,column_name,missing_values_threshold):
        
        if np.count_nonzero(data[column_name] == np.nan) >= int(missing_values_threshold*len(data)):
            
            data.drop([column_name],axis=1,inplace=True)
            
            return None
            
        elif isinstance(data[column_name][0],str):
            
            return 'string'
        
        else:
            
            return 'float64'
        
        
        
    def fillna(self,data,datatype_dict):
        
        numpy_array_list = list()
        
        for col in datatype_dict.keys():
            
            if datatype_dict[col] == 'string':
                    
                imputer = SimpleImputer(strategy='most_frequent')
                
                column_data = np.array(data[col])
                
                column_data = column_data.reshape(column_data.shape[0],1)
                
                imputer.fit(column_data)
                
                column_data = imputer.transform(column_data)
            
                data[col] = column_data.reshape(column_data.shape[0],1)
                
                data[col].replace(to_replace=data[col].unique(),value=list(range(0,len(data[col].unique()))),inplace=True)
                
                numpy_array_list.append(np.eye(len(data[col].unique()),len(data[col].unique()))[data[col]])
                
            elif datatype_dict[col] == 'float64':
                
                imputer = SimpleImputer(strategy='mean')
                
                column_data = np.array(data[col])
                
                column_data = column_data.reshape(column_data.shape[0],1)
                
                imputer.fit(column_data)
                
                column_data = imputer.transform(column_data)
                
                data[col] = column_data.reshape(column_data.shape[0],1)
                
                numpy_array_list.append(np.array(data[col]).reshape(data[col].shape[0],1))
                
            numpy_data = np.concatenate(tuple(numpy_array_list),axis=1)
            
        return numpy_data
    
    
    
    def apply_PCA(self,numpy_transformed_data,explained_variance_fraction):
        
        pca = PCA(n_components=numpy_transformed_data.shape[1])
        
        pca.fit(numpy_transformed_data)
        
        eig_vals_total = np.sum(pca.explained_variance_)
        
        eig_vals_sum = 0
        
        vecs_count = 0
        
        for explained_variance in pca.explained_variance_:
            
            if (eig_vals_sum/eig_vals_total) > explained_variance_fraction:
                
                break
                
            eig_vals_sum = eig_vals_sum + explained_variance
            
            vecs_count = vecs_count + 1
            
        pca = PCA(n_components=vecs_count)
        
        dim_reduced_data = pca.fit_transform(numpy_transformed_data)
        
        return dim_reduced_data
    
    
    
    def determine_distribution_type(self,feat_vecs):
        
        dist_params_list = []
        
        for index in range(feat_vecs.shape[1]):
        
            sample_mean = np.mean(feat_vecs[:,index])
        
            sample_std = np.std(feat_vecs[:,index])
        
            sample_stat = np.sqrt(np.mean(feat_vecs[:,index]**2))
        
            neg_log_func_gaussian = -np.sum(np.log(s.norm.pdf(sample_mean,sample_std,feat_vecs[:,index])))
        
            neg_log_func_rayleigh = -np.sum(np.log(s.norm.pdf(sample_stat,feat_vecs[:,index])))
        
            if neg_log_func_gaussian < neg_log_func_rayleigh:
            
                dist_params_list.append(("normal",(sample_mean,sample_std)))
        
            else:
            
                dist_params_list.append(("rayleigh",sample_stat))
                
        return dist_params_list


# In[86]:


if __name__ == "__main__":
    
    obj = custom_data_reader()


# In[87]:


obj = custom_data_reader()


# In[88]:


data_results = obj.read_csv("data.csv","diagnosis",0.8,True,0.99)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




