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


folder_path= 'C:~'#The address of the first folder containing RPT data

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


        df_raw3=df_raw[['TotCycle','StepNo','Voltage(V)', 'Current(A)']]
        df_raw3=df_raw3.astype(float)
        df_raw3=df_raw3.astype({'TotCycle':'int'})
        df_raw3=df_raw3.astype({'StepNo':'int'})
        
        df_1_dr=df_raw3[(df_raw3['TotCycle']==30) & (df_raw3['StepNo']>=33)]
        df_1_dr.reset_index(drop=True, inplace=True)
        columns_data=['Current', 'Voltage']
        df_1_dr2=pd.DataFrame(columns=columns_data)
        df_1_dr2['Cyc']=[1,2,3,4,5,6]
        df_1_dr2=df_1_dr2.set_index('Cyc')
        df_1_dr2.reset_index(drop=True, inplace=True)
        for i in [17,18,19,20,21,22]:
            df_new1=df_1_dr[(df_1_dr['StepNo']==(2*i)-1)]
            df_1_dr2.iloc[i-17,0]=df_new1.iloc[-1,3]
            df_1_dr2.iloc[i-17,1]=df_new1.iloc[-1,2]
        from sklearn.linear_model import LinearRegression 
        X=df_1_dr2['Current'].values.reshape(-1,1)
        y=df_1_dr2['Voltage'].values
        model=LinearRegression().fit(X,y)
        beta_1=model.coef_[0]
        
        data_folder2='C:~'#The directory path of the folder where the data-saving file is located.
        excel_file=data_folder2+'DCIR@SOC50.xlsx'#The Excel file address to save the data of DCIR@SOC50 corresponding to each cell's RPT
        wb=xw.Book(excel_file)
        sht=wb.sheets['DCIR']
        sht.range((row+2,num+1)).value= beta_1
        wb.save(excel_file)
        wb.app.quit() #Extracting DCIR at SOC50
        


        df_raw4=df_raw[['TotCycle','StepNo','Char. Cap.(Ah)','Dischar. Cap.(Ah)']]
        df_raw4=df_raw4.astype(float)
        df_raw4=df_raw4.astype({'TotCycle':'int'})
        df_raw4=df_raw4.astype({'StepNo':'int'})

        df_1_q=df_raw4[(df_raw4['TotCycle']==3) & (df_raw4['StepNo']==6)]
        ch1=df_1_q.iloc[-1,2]

        df_2_q=df_raw4[(df_raw4['TotCycle']==3) & (df_raw4['StepNo']==7)]
        dc1=df_2_q.iloc[-1,3]

        df_3_q=df_raw4[(df_raw4['TotCycle']==5) & (df_raw4['StepNo']==11)]
        ch2=df_3_q.iloc[-1,2]

        df_4_q=df_raw4[(df_raw4['TotCycle']==5) & (df_raw4['StepNo']==12)]
        dc2=df_4_q.iloc[-1,3]


        data_folder3='C:~'#The directory path of the folder where the data-saving file is located.
        excel_file3=data_folder3+'0.2C & 1C_Capacity.xlsx'#The Excel file address to save the data of 0.2C & 1C capacity corresponding to each cell's RPT
        wb=xw.Book(excel_file3)
        sht=wb.sheets['0.2C']


        sht.range((row+3,num*2)).value= ch1
        sht.range((row+3,num*2+1)).value= dc1

        sht2=wb.sheets['1C']

           

        sht2.range((row+3,num*2)).value= ch2
        sht2.range((row+3,num*2+1)).value= dc2
        wb.save(excel_file3)
        wb.app.quit()#Extracting 0.2C and 1C capacity
            
        df_raw2=df_raw[['TotCycle','StepNo','Voltage(V)']]
        df_raw2=df_raw2.astype(float)
        df_raw2=df_raw2.astype({'TotCycle':'int'})
        df_raw2=df_raw2.astype({'StepNo':'int'})
        
        df_ch_gi1=df_raw2[(df_raw2['TotCycle']>=7) & (df_raw2['TotCycle']<=15)]
        df_ch_gi1.reset_index(drop=True, inplace=True)
        columns_data=['Cyc', 'DCIR']
        df_ch_gi2=pd.DataFrame(columns=columns_data)
        df_ch_gi2['Cyc']=[7,8,9,10,11,12,13,14,15]
        df_ch_gi2=df_ch_gi2.set_index('Cyc')
        for i in df_ch_gi2.index:
            df_new1=df_ch_gi1[(df_ch_gi1['TotCycle']==i)&(df_ch_gi1['StepNo']==20)]
            dc=(df_new1.iloc[1,2]-df_new1.iloc[0,2])/0.6
            df_ch_gi2.loc[i,'DCIR']=dc
        DCIR=np.array(df_ch_gi2['DCIR']).reshape(-1,1) 

        
        data_folder='C:~'#The directory path of the folder where the data-saving file is located.
        excel_file=data_folder+'DCIR_from_GITT.xlsx'#The Excel file address to save the data of DCIR vs SOC corresponding to each cell at EOL
        wb=xw.Book(excel_file)
        sht=wb.sheets['RPT'+str(row)]
        sht.range((2,num+1)).value= DCIR
        wb.save(excel_file)
        wb.app.quit() #Extracting DCIR vs SOC0~80 from GITT
 
        
        
        
        

       
        row=row+1
    num=num+1


# In[ ]:




