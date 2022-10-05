# -*- coding: utf-8 -*-
"""
Created on Thu May  5 16:01:40 2022

@author: suhlr
"""
import requests
from decouple import config
import getpass
import maskpass
import os
from utilsAPI import getAPIURL

API_URL = getAPIURL()

#%% Get token 
def getToken(saveEnvPath=None):
           
    if 'API_TOKEN' not in globals():
    
        try: # look in environment file
            token = config("API_TOKEN")              
            
        except:
            try:
                # If spyder, use maskpass
                isSpyder = 'SPY_PYTHONPATH' in os.environ
                isPycharm = 'PYCHARM_HOSTED' in os.environ
                print('Login with credentials used at app.opencap.ai.\nVisit the website to make an account if you do not have one.\n')
                
                if isSpyder:
                    un = maskpass.advpass(prompt="Enter Username:\n",ide=True)
                    pw = maskpass.advpass(prompt="Enter Password:\n",ide=True)
                elif isPycharm:
                    print('Warning, you are in Pycharm, so the password will show up in the console.\n To avoid this, run createAuthenticationEnvFile.py from the terminal,\nthen re-open PyCharm.')
                    un = input("Enter username:")
                    pw = input("Enter password (will be shown in console):")
                else:
                    un = getpass.getpass(prompt='Enter Username: ', stream=None)
                    pw = getpass.getpass(prompt='Enter Password: ', stream=None)
                
                data = {"username":un,"password":pw}
                resp = requests.post(API_URL + 'login/',data=data).json()
                token = resp['token']
                
                print('Login successful.')
                
                if saveEnvPath is not None:
                    envPath = os.path.join(saveEnvPath,'.env')
        
                    f = open(envPath, "w")
                    f.write('API_TOKEN="' + token + '"')
                    f.close()
                    print('Authentication token saved to '+ envPath + '. DO NOT CHANGE THIS FILE NAME. If you do, your authentication token will get pushed to github. Restart your terminal for env file to load.')

            except:
                raise Exception('Login failed.')
        
        global API_TOKEN
        API_TOKEN = token
    else:
        token = API_TOKEN
    
    return token
