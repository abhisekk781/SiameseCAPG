# =========================================================================================================================
#   File Name           :   predictions.py
# -------------------------------------------------------------------------------------------------------------------------
#   Purpose             :   Purpose of this script is to read files from input directory and generate predictions in output dir 
#   Author              :   Abhisek Kumar
#   Co-Author           :   
#   Creation Date       :   28-August-2020
#   History             :
# -------------------------------------------------------------------------------------------------------------------------
#   Date            | Author                        | Co-Author                                          | Remark
#   28-August-2020    | Abhisek Kumar                                         | Initial Release
# =========================================================================================================================
# =========================================================================================================================
# Import required Module / Packages
# -------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from fbprophet import Prophet
import datetime
import time
import sys
from datetime import date
from dateutil.relativedelta import relativedelta
import os, shutil
import warnings
from pyxlsb import convert_date
warnings.filterwarnings('ignore')
import random
random.seed(35)
try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser

# -------------------------------------------------------------------------------------------------------------------------
# Read the configuration file
config = ConfigParser()
config.read('emon_config.ini')


# -------------------------------------------------------------------------------------------------------------------------
# Set all required global variables
INPUT_PATH = config['FILE']['INPUT_PATH']
DATA_PATH = config['FILE']['DATA_PATH']
FAILURE_PATH = config['FILE']['FAILURE_PATH']
OUTPUT_PATH = config['FILE']['OUTPUT_PATH']
DATE = config['COLUMNS']['Date']
NAME = config['COLUMNS']['Name']
ACTUALS_TOTAL_COST = config['COLUMNS']['Actuals_Total_Cost']
ACTUALS_TOTAL_REV = config['COLUMNS']['Actuals_Total_Rev']
BUDGET_TOTAL_COST = config['COLUMNS']['Budget_Total_Cost']
BUDGET_TOTAL_REV = config['COLUMNS']['Budget_Total_Rev']
EAC_TOTAL_COST = config['COLUMNS']['EAC_Total_Cost']
EAC_TOTAL_REV = config['COLUMNS']['EAC_Total_Rev']
RTBD_TOTAL_COST = config['COLUMNS']['RTBD_Total_Cost']
RTBD_TOTAL_REV = config['COLUMNS']['RTBD_Total_Rev']
ACTUALS_CM_PERC = config['COLUMNS']['Actuals_CM_Perc']
BUDGET_CM_PERC = config['COLUMNS']['Budget_CM_Perc']
EAC_CM_PERC = config['COLUMNS']['EAC_CM_Perc']
RTBD_CM_PERC = config['COLUMNS']['RTBD_CM_Perc']
LOGGER = config['DEBUG']['log']
LOOK_BACK = config['INPUT']['lookbackperiod']
FUTURE = config['INPUT']['future']
rowSkip = config['EXTRAS']['ROW_SKIP']
if len(config['SHEET']['SHEET_NAME'])>0:
    SHEET_NAME = config['SHEET']['SHEET_NAME']
else: SHEET_NAME = 0
    



# -------------------------------------------------------------------------------------------------------------------------
# Configure the logger file
import logging
logging.basicConfig(filename=LOGGER,
                    format='[%(levelname)s][%(asctime)s]::%(message)s', 
                    datefmt='%d/%m/%Y %I:%M:%S')

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------------------------------
# Setting additional global variables
colArray = [DATE,NAME,ACTUALS_TOTAL_COST,ACTUALS_TOTAL_REV,ACTUALS_CM_PERC,BUDGET_TOTAL_COST,BUDGET_TOTAL_REV,BUDGET_CM_PERC,
           RTBD_TOTAL_COST,RTBD_TOTAL_REV,RTBD_CM_PERC,EAC_TOTAL_COST,EAC_TOTAL_REV,EAC_CM_PERC]

#Filter None values
colArray = list(filter(None,colArray))

ignoreFields = [NAME]
ignoreFields2 = [NAME,DATE]


###########################################################################################################################
# Author        : Abhisek Kumar
# Co-Author     : 
# Modified      :  
# Reviwer       :                                                                                         
# Functionality : Checks for instance of string and replaces it with NaN                                                               
###########################################################################################################################
def check_missing(value):
    try:
        return np.where(isinstance(value,str),np.NaN,value)
    except Exception as e:
        raise(e)     

        
###########################################################################################################################
# Author        : Abhisek Kumar
# Co-Author     : 
# Modified      :  
# Reviwer       :                                                                                         
# Functionality : loop through all columns and check for string instance and replaces it with NaN, drops missing rows,
#                 converts string data into date-time format
###########################################################################################################################
def adjust_missing(pDf,columnArray,ignoreFields):
    try:
        pDf = pDf[columnArray]
        columns = [col for col in columnArray if col not in ignoreFields]
        for col in columns:
            pDf[col] = pDf[col].apply(lambda s: check_missing(s))
        pDf.dropna(how='any',inplace=True)#.reset_index(drop=True)
        pDf[DATE] = pDf[DATE].apply(lambda x: convert_date(x))
        pDf = pDf[pDf[DATE].notnull()]
        #pDf[DATE] = pDf[DATE].apply(lambda x:pd.to_datetime(x))
        pDf[DATE] = pd.to_datetime(pDf[DATE])
        return pDf
    except Exception as e:
        raise(e)
        


###########################################################################################################################
# Author        : Abhisek Kumar
# Co-Author     : 
# Modified      :  
# Reviwer       :                                                                                         
# Functionality : Generates list of past months from maximum available date in the data.xlsx file                                                       
###########################################################################################################################        
def getPrevMonths(pDf,dateCol,pastMonths):
    try:
        #pDf[dateCol] = pDf[dateCol].apply(lambda x: convert_date(x))
        maxDate = max(pDf[dateCol].values)
        maxDate = pd.to_datetime(maxDate)
        #print(maxDate)
        allPrevDates = []
        allPrevDates.append(maxDate.strftime('%Y-%m-%d'))
        for i in range(0,pastMonths):
            prevMonth = maxDate+relativedelta(months=-i)
            allPrevDates.append(prevMonth.strftime('%Y-%m-%d'))
        return allPrevDates
    except Exception as e:
        raise(e)
        
        
###########################################################################################################################
# Author        : Abhisek Kumar
# Co-Author     : 
# Modified      :  
# Reviwer       :                                                                                         
# Functionality : filters all the active engagements                                                               
###########################################################################################################################     
def filter_names(pDf,date,dateCol,nameCol):
    try:
        tempDF = pDf.groupby(dateCol).get_group(date)
        tempNames = list(set(tempDF[nameCol].values))
        return tempNames  
    except Exception as e:
        raise(e)
        
              
###########################################################################################################################
# Author        : Abhisek Kumar
# Co-Author     : 
# Modified      :  
# Reviwer       :                                                                                         
# Functionality : Generates future predictions for all engagments                                                               
###########################################################################################################################
def get_predictions(pDf,nameCol,engName,dateCol,lookback,futureMonths,columnArray):
    try:
        gDF = pDf.groupby(NAME).get_group(engName).groupby([dateCol]).mean().iloc[-int(lookback):,:]
        futureDates = list(gDF.index)

        for i in range(1,int(futureMonths)+1):
            futureDates.append(gDF.index[-1] + relativedelta(months=+i))
        future = pd.DataFrame({'ds':futureDates})

        predictions = pd.DataFrame()
        predictions[DATE] = future['ds']
        prefix = ['Pred ','Status', 'Original', 'Predicted']

        for i in columnArray:
            fb = pd.DataFrame()
            fb['y'] = gDF[i]
            fb['ds'] = gDF.index
            model = Prophet()
            forecast = model.fit(fb).predict(future)
            predictions[prefix[0]+i] = forecast['yhat']
            predictions[prefix[0]+i+' Upper'] = forecast['yhat_upper']
            predictions[prefix[0]+i+' Lower'] = forecast['yhat_lower']

        first = predictions.iloc[:-int(futureMonths),:]
        first[prefix[1]] = prefix[2]
        last = predictions.iloc[-int(futureMonths):,:]
        last[prefix[1]] = prefix[3]  
        reset_DF = gDF.reset_index()
        #
        final_DF = reset_DF.drop(dateCol,axis = 1)
        merged_df = pd.concat([first,final_DF],axis = 1)
        merged_df_final = pd.concat([merged_df,last])
        merged_df_final.insert(1,nameCol,[engName for i in range(len(future['ds']))],True)
        return merged_df_final
    except Exception as e:
        raise(e)
        
        
###########################################################################################################################
# Author        : Abhisek Kumar
# Co-Author     : 
# Modified      : 
# Reviwer       :                                                                                         
# Functionality : Reading and appending files from input directory                                                               
###########################################################################################################################       
        
        
def Filelist(pDir,rowSkip = rowSkip, sheetName = SHEET_NAME):
    try:
        pData = pd.DataFrame()
        pFiles, pAppendData = [],[]
        for file in os.listdir(pDir):
            pFiles.append(file)
           
        if len(pFiles) > 0:      
            for file in os.listdir(os.path.join(pDir)):
                pDataFile = pd.read_excel(os.path.join(pDir, file),sheet_name = sheetName,engine='pyxlsb')
                #Skip rows to get the relevant column rows
                if len(rowSkip) > 0 and int(rowSkip)>=1:
                    pDataFile = pDataFile.rename(columns=pDataFile.iloc[int(rowSkip)-1]).drop(pDataFile.index[0:int(rowSkip)]).reset_index(drop=True)
                pAppendData.append(pDataFile)
            pData = pd.concat(pAppendData)
         
    except Exception as e:
        raise(e)    
    return pData, pFiles
        

###########################################################################################################################
# Author        : Abhisek Kumar
# Co-Author     : 
# Modified      :  
# Reviwer       :                                                                                         
# Functionality : Define the main function                                                              
###########################################################################################################################
        
def main():
    
    try:
            #Read input file and combine with existing archive file
        inputData, inputFiles = Filelist(INPUT_PATH)
        #write one for data
        if len(os.listdir(DATA_PATH)) > 0:
            archiveList = [f for f in os.listdir(DATA_PATH) if f not in inputFiles]
            if len(archiveList) > 0:
                pAppendArchiveData = []
                for arfile in archiveList:
                    pArchiveData = pd.read_excel(os.path.join(DATA_PATH, arfile),sheet_name = SHEET_NAME ,engine='pyxlsb')
                    if len(rowSkip) > 0 and int(rowSkip)>=1:
                        pArchiveData = pArchiveData.rename(columns=pArchiveData.iloc[int(rowSkip)-1]).drop(pArchiveData.index[0:int(rowSkip)]).reset_index(drop=True)
                    pAppendArchiveData.append(pArchiveData)
                archiveData = pd.concat(pAppendArchiveData)
                inputData = pd.concat([archiveData,inputData])
              
        
           #Start the preprocessing
        DF = adjust_missing(inputData,colArray,ignoreFields)
        
            #Converting all columns to numeric
        reqCols = [col for col in colArray if col not in ignoreFields2]
        DF[reqCols] = DF[reqCols].apply(pd.to_numeric, errors='coerce')
          
            #Getting all active engagements
        allnames = list(map(lambda s: filter_names(DF,s,DATE,NAME),getPrevMonths(DF,DATE,int(LOOK_BACK))))
        final_names = list(set(allnames[0]).intersection(*allnames[1:]))
        
            #generating predictions for all active engagements
        predictionsAll = list(map(lambda s: get_predictions(DF,NAME,s,DATE,LOOK_BACK,FUTURE,reqCols), final_names))
        results = pd.concat(predictionsAll)

            #Exporting predictions in .xlsx format
        print('Creating output file...')
        output_file_name = str(datetime.datetime.now().strftime('%d-%m-%Y_%H_%M_%S_')) + 'output.xlsx'
        results.to_excel(os.path.join(OUTPUT_PATH,output_file_name), index = False)
        print('Output file created in output directory!')  
            #Move the files to data directory
        path = INPUT_PATH
        moveto = DATA_PATH
        files = os.listdir(path)
        files.sort()
        for f in files:
            src = path+f
            dst = moveto+f
            shutil.move(src,dst)
        print('Successful: Moving file(s) to data directory...')
        time.sleep(2.5)
    except Exception as e:
        print('Failed: Moving file(s) to failure directory...')
        path = INPUT_PATH
        moveto = FAILURE_PATH
        files = os.listdir(path)
        files.sort()
        for f in files:
            src = path+f
            dst = moveto+f
            shutil.move(src,dst)
        logger.exception(e)
        time.sleep(2.5)