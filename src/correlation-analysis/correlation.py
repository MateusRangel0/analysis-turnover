# importing pandas module 
import pandas as pd 

# importing regex module 
import re 
	
# making data frame 
data = pd.read_csv("2lovemondays_turnover.csv") 

correlation = data.corr(method="pearson")

correlation.to_csv("plot.csv")

print(data.corr(method="pearson"))