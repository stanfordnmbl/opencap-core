# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 13:37:39 2022

@author: suhlr
"""
from decouple import config

def getAPIURL():
    if 'API_URL' not in globals():
        global API_URL
        try: # look in environment file
            API_URL = config("API_URL")
        except: # default
            API_URL = "https://api.opencap.ai/"
    
    if API_URL[-1] != '/':
        API_URL= API_URL + '/'

    return API_URL