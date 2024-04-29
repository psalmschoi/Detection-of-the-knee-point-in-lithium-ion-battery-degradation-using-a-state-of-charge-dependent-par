#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xlsxwriter
import xlwings as xw
import os 


# In[2]:


folder_path= 'C:~'#The address of the first folder containing RPT data'

def get_subfolders(folder_path):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    return sorted(subfolders, key=lambda x: (int(os.path.basename(x).split('.')[0]), x.split('.')[1]))


subfolders = get_subfolders(folder_path)

num=1 #a variable used for storing data in specific location in excel

for path_2 in subfolders:   
    file_lis=os.listdir(path_2) # The second folder including RPT csv files of correponding cell name
   
   
    for i in range(len(file_lis)):
        file_lis[i]=file_lis[i][:-4]
    
    file_lis = list(map(int, file_lis))
    file_lis.sort()#Sort the names of  RPT csv files in ascending order
    file_lis = list(map(str, file_lis))
        
        
    row=0 #a variable used for storing data in specific location in excel
    for x in file_lis:
        df_raw=pd.read_csv(path_2+'/'+x+'.csv',encoding='cp949',header=0)


             
        df_raw7=df_raw[['TotCycle','StepNo','Voltage(V)','Char. Cap.(Ah)']]
        df5=df_raw7[(df_raw7['TotCycle']==5) & (df_raw7['StepNo']==11)]
        df6=df5[['Voltage(V)','Char. Cap.(Ah)']]
        df6=df6[df6['Voltage(V)']<4.2]
        df6.reset_index(drop=True, inplace=True)
        df16=df6.drop_duplicates(['Voltage(V)']) 
        from scipy import interpolate
        x_ori = df16['Voltage(V)']
        y_ori = df16['Char. Cap.(Ah)']
        x=np.array(x_ori)
        y=np.array(y_ori)
        f1 = interpolate.interp1d(x,y, kind='quadratic')
        x_new = np.linspace(4.195,3,num=1000,endpoint=True)
        y_new = f1(x_new)
        df_raw8=df_raw[['TotCycle','StepNo','Voltage(V)','Char. Cap.(Ah)']]
        df10=df_raw8[(df_raw8['TotCycle']==3) & (df_raw8['StepNo']==6)]
        df11=df10[['Voltage(V)','Char. Cap.(Ah)']]
        df11=df11[df11['Voltage(V)']<4.2]
        df11.reset_index(drop=True, inplace=True)
        df12=df11.drop_duplicates(['Voltage(V)'])
        x_ori1 = df12['Voltage(V)']
        y_ori1 = df12['Char. Cap.(Ah)']
        x1=np.array(x_ori1)
        y1=np.array(y_ori1)
        f2 = interpolate.interp1d(x1,y1, kind='quadratic')
        x_new1= np.linspace(4.195,3,num=1000,endpoint=True)
        y_new1 = f2(x_new)
        delta_y=y_new1-y_new
        columns_data=['Voltage','Delta Q']
        df_new=pd.DataFrame(columns=columns_data)
        
        df_new['Delta Q']=delta_y
        df_new['Voltage']=x_new1
        df_new2=df_new[df_new['Voltage']>=3.3]
        var=df_new2['Delta Q'].var()
        df_final=df_new.set_index('Voltage')
        data_folder='C:~'#The directory path of the folder where the data-saving file is located.
        excel_file=data_folder+'1C-0.2C_delta_Q(V).xlsx'#The Excel file address to save the data of Q0.2C-1C(V) corresponding to each cell's RPT
        wb=xw.Book(excel_file)
        sht=wb.sheets['RPT'+str(row)]
        sht.range((3,num+1)).value= np.array(df_final['Delta Q']).reshape(-1,1)
        sht.range((3,1)).value= np.array(df_new['Voltage']).reshape(-1,1)
        wb.save(excel_file)
        wb.app.quit() 

        data_folder='C:~'#The directory path of the folder where the data-saving file is located.
        excel_file=data_folder+'Var_1C-0.2C_delta_Q(V).xlsx'#The Excel file address to save the data of Var (Q0.2C-1C(V)) corresponding to each cell's RPT
        wb=xw.Book(excel_file)
        sht=wb.sheets['Variance']
        sht.range((row+3,num+1)).value=var
        wb.save(excel_file)
        wb.app.quit() 
        
       
        

       
        row=row+1
    num=num+1


# In[ ]:




