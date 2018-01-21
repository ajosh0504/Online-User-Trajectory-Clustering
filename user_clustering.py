"""
Created on Mon Dec  4 20:28:53 2017

@author: apoorva
"""
from __future__ import print_function
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords 
import numpy as np
import re
import itertools
from collections import Counter
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import cPickle as pickle
import math

test_set=pd.read_csv('posts_12_4.txt', sep= '\t', header= None, encoding='latin1')
user_dict={}
user_dict_month={}
user_dict_days={}
month={'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
month_days={1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
res_vect= np.load('res_5.npy')
for i in range(0,2854280):
    print(i)
    tmp=test_set.iloc[i][2]
    if not(math.isnan(tmp)):
        date_string=test_set.iloc[i][6]
        if isinstance(date_string, basestring):
            sum_days=0
            try:
                date_reg=re.match( r'\A.*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*(\d*),?\s*(\d*)', date_string, re.I)
                m = month[date_reg.group(1)]
                d = int(date_reg.group(2))
                y = int(date_reg.group(3))
                for k in range(2002,y):
                    if k%4:
                        sum_days=sum_days + 365
                    else:
                        sum_days = sum_days + 366
                sum_days = sum_days + d
                for k in range(1,m-1):
                    if y%4==0 and k==2: 
                        sum_days = sum_days +29
                    else:    
                        sum_days= sum_days + month_days[k]
            except: 
                continue
            if tmp in user_dict_days.keys():
                
                #user_dict[tmp].append([m,d,y])
                user_dict_days[tmp].append((sum_days,res_vect[i]))
            else:
                #user_dict[tmp]= [[m,d,y]]
                user_dict_days[tmp]=[(sum_days,res_vect[i])]

with open('user_dict_days.pickle', 'wb') as fp:
  pickle.dump(user_dict_days, fp)

def num_posts(lst,begin,end,last_day):
    count=np.zeros((5))
    if end == last_day:
        for i in lst:
            if i[0]>=begin and i[0]<=end:
                count= count+i[1]
    else:        
        for i in lst:
            if i[0]>=begin and i[0]<end:
                count= count+i[1]
    return(count)   
     
user_dict_days = pickle.load(open('user_dict_days_new.pickle','rb')) 
for i in user_dict_days.keys():
    user_dict_days[i].sort(key=lambda x: x[0]) 
    
user_activity={}
var=28
month_nopost={}
month_zeros={}
active_users={}
for i in sorted(user_dict_days.keys()):
    temp=user_dict_days[i]     
    first_day=temp[0][0]
    last_day=temp[len(temp)-1][0]
    num_weeks= int((last_day-first_day)/var)
    res_dict={}
    for k in range(0,num_weeks):
        res_dict[k+1]=num_posts(temp,first_day+var*k,first_day+var*(k+1),last_day)
    if (first_day-last_day)%var:
        res_dict[num_weeks+1]= num_posts(temp,first_day+var*num_weeks,last_day,last_day)
    if first_day == last_day:
        res_dict[1]= num_posts(temp,first_day,last_day,last_day)
    user_activity[i]=res_dict
    
    X = 12 # Same X from email
    
    if len(res_dict) >= X:
        # Check how many inactive months
        zero_count = 0 
        for j in range(1, X+1): # check up to and including X month in res_dict
            try:
                if np.array_equal(res_dict[j], np.zeros((5))):
                    zero_count += 1
            except:
                break
        month_nopost[i] = zero_count
        # Check how many 'zero' months
        zero_count = 0 
        for j in range(1, X+1): # check up to and including X month in res_dict
            try:
                if res_dict[j][4] > 0 and np.array_equal(res_dict[j][0:4], np.zeros((4))):
                    zero_count += 1
            except:
                break
        month_zeros[i] = zero_count
            
for i in month_nopost.keys():
    if month_nopost[i] < 6:
        active_users[i] = []
        for j in range(1, 13):
            active_users[i].append(user_activity[i][j])  
ind={} 
i=0           
for j in sorted(active_users.keys()):
    ind[i]=j
    i+=1
           
data_clust=[]
for i in sorted(active_users.keys()):
    temp=[]
    for j in list(active_users[i]):
        temp.append(j)
        tmp= [x for sub_list in temp for x in sub_list]
    data_clust.append(tmp)
data_clust= np.array(data_clust)
np.savetxt('data_clust.txt',data_clust,fmt='%.0f',delimiter=' ')       
    
#month_count={}
#for i in range(0,X+1):
#    month_count[i]=0
#
#for i in month_nopost.values():
#    month_count[i]+=1
#
#plt.bar(month_count.keys(), month_count.values(),1.0, color='b')     
#plt.show()  
#
#for i in sorted(month_count.keys()):
#    print('Month '+str(i)+': '+str(month_count[i]))             
        
    
for i in sorted(user_activity.keys()):
    print('User '+ str(i) + ': ', end=' ')
    for j in user_activity[i].keys():  
        print('Month ' + str(j) + '=' + str(user_activity[i][j]), end=' ')
    print('')    
       
#Visualizing Weekly distibution
user_dict_days = pickle.load(open('user_dict_days.pickle', 'rb'))
weeks_dict={}
for p in user_dict_days.values():
    x=[]
    for l in p:
        x.append(l[0])
    span= math.ceil((max(x)-min(x))/7.0)
    if span in weeks_dict.keys():
        weeks_dict[span] += 1
    else:
        weeks_dict[span] = 1        
s_max = max(weeks_dict.keys())
s_min = min(weeks_dict.keys())
#weeks_dict[0]=0
for i in range(int(s_min),int(s_max)):
    if i not in weeks_dict.keys():
        weeks_dict[i]=0
sum_all=sum(weeks_dict.values())
morethanxweeks={} 
temp_sum=0       
for i in range(0,142):
    if i>0:
        temp_sum=temp_sum + weeks_dict[i-1]
    morethanxweeks[i]=sum_all-temp_sum  
for x in sorted(morethanxweeks.keys()):
    print('Week '+str(x)+'-'+str(morethanxweeks[x]))          
plt.bar(morethanxweeks.keys(), morethanxweeks.values(),1.0, color='b')     
plt.show()   
