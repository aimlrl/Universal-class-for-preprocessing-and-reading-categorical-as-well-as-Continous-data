#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd

import numpy as np

from sklearn.impute import SimpleImputer

from sklearn.decomposition import PCA


# In[5]:


class custom_data_reader():
    
    
    def read_csv(self,file_name,missing_values_threshold,apply_dim_reduction,explained_variance_fraction):
        
        datatype_dict = {}
        
        data = pd.read_csv(file_name)
        
        data.dropna(axis=1,inplace=True)
        
        for col in data.columns:
            
            datatype_dict[col] = self.determine_data_type(data,col,missing_values_threshold)
            
        filled_data = self.fillna(data,datatype_dict)
        
        if apply_dim_reduction == True:
        
            dim_reduced_data = self.apply_PCA(filled_data,explained_variance_fraction)
            
            return dim_reduced_data
        
        else:
            
            return filled_data
            
            
        
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
        
        ratio_sum = 0
        
        vecs_count = 0
        
        for explained_variance in pca.explained_variance_ratio_:
            
            if ratio_sum > explained_variance_fraction:
                
                break
                
            ratio_sum = ratio_sum + explained_variance_fraction
            
            vecs_count = vecs_count + 1
            
        pca = PCA(n_components=vecs_count)
        
        dim_reduced_data = pca.fit_transform(numpy_transformed_data)
        
        return dim_reduced_data


# In[6]:


if __name__ == "__main__":
    
    obj = custom_data_reader()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[86]:





# In[ ]:





# In[ ]:




