# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:52:26 2020

@author: lhatc
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

maxAge = 109;

###############################################################################
def extractDeathByYear(filename='deathByAgeSingleYear.txt'):

    rows = [] 
    with open(filename,'r',encoding="ASCII") as csvfile:
        # creating a csv reader object 
        csvreader = csv.reader(csvfile, delimiter='\t') 
          
        # extracting field names through first row 
        fields = next(csvreader) 
      
        # extracting each data row one by one 
        for row in csvreader: 
            rows.append(row) 
     
        return rows, fields;

###############################################################################
def calcConditionalCDF(cdf, age=0):

    if age > 0:
        condCDF = (cdf - cdf[age-1])/(1-cdf[age-1]);
    else:
        condCDF = cdf;
    condCDF = condCDF[age:,];
    condPDF = np.append(condCDF[0],np.diff(condCDF));
    return condCDF, condPDF;

###############################################################################
def calcConditionalLifeExpect(cdf):

    count = 0;
    lifeExpectancy = np.zeros(maxAge-1)
    for age in range(0,maxAge-1):
        condCDF, condPDF = calcConditionalCDF(cdf,age);
        ageLeft = np.array([ii for ii in range(age,maxAge)]);
        lifeExpectancy[count] = np.sum(condPDF*ageLeft);
        count = count + 1;
        
    return lifeExpectancy;

###############################################################################
rows, fields = extractDeathByYear();

ageIndex = fields.index("Single-Year Ages Code");
rateIndex = fields.index("Crude Rate");

ageArray = np.array([float(row[ageIndex]) for row in rows]);
probDeathArray = np.array([float(row[rateIndex])/100000 for row in rows]);

# plot 1 - rate of death by age
fig = plt.figure();
ax = fig.add_subplot(1, 1, 1);
ax.plot(ageArray, probDeathArray,".-");
ax.set_xlabel("Age (years)");
ax.set_ylabel("Rate of death");
ax.grid();

# fit exponential curve
logYData = np.log(probDeathArray);
curveFit = np.polyfit(ageArray, logYData, 2);

ageArrayExtend = np.array(range(85,maxAge));
ageArrayTotal = np.append(ageArray,ageArrayExtend);

probDeathExtend = np.exp(curveFit[2]) * np.exp(curveFit[1]*ageArrayExtend) * np.exp(curveFit[0]*ageArrayExtend**2);
probDeathTotal = np.append(probDeathArray, probDeathExtend);
probDeathTotal[-1] = 1; # force the last data point to probability 1

fig = plt.figure();
ax = fig.add_subplot(1, 1, 1);
ax.plot(ageArray, probDeathArray, "o-")
ax.plot(ageArrayTotal, probDeathTotal,'.-')
ax.set_xlabel("Age (years)");
ax.set_ylabel("Rate of death");
ax.grid();
ax.legend(("Original data","Extrapolated Data"));


##
#fig = plt.figure();
#ax = fig.add_subplot(1, 1, 1);
temp1 = np.append(1,np.cumprod((1-probDeathTotal)));
temp1 = temp1[0:temp1.shape[0]-1]
temp2 = probDeathTotal;
pdf = temp1*temp2;
fig = plt.figure();
ax1 = fig.add_subplot(2, 1, 1);
ax1.plot(ageArrayTotal,pdf,'.-')
ax1.set_ylabel(r'${p_a(a)}$');
ax1.set_xlabel("Age (years)");
ax1.grid();

cdf = np.cumsum(pdf);
ax2 = fig.add_subplot(2, 1, 2);
ax2.plot(ageArrayTotal,cdf,'.-')
ax2.set_xlabel("Age (years)");
ax2.set_ylabel(r'${F_a(a)}$');
ax2.grid();

lifeExpectancy = (np.sum(pdf*ageArrayTotal))
print(("Expected value of age of death = ", lifeExpectancy))

ax1.plot((lifeExpectancy,lifeExpectancy),(0,max(pdf)),'r--')


cdf36, pdf36 = calcConditionalCDF(cdf,36);
cdf78, pdf78 = calcConditionalCDF(cdf,78);

fig = plt.figure();
ax3 = fig.add_subplot(2, 1, 1);
ax4 = fig.add_subplot(2, 1, 2);

labelArray = list();
for age in np.array((10,30,60,78)):
    currCDF, currPDF = calcConditionalCDF(cdf,age);
    ax3.plot(np.array(range(age,maxAge)),currCDF,'.-');
    ax4.plot(np.array(range(age,maxAge)),currPDF,'.-');
    labelArray.append(("Age = ", str(age)));
 
ax3.grid();
labelString = r'$F_{a|a_0}(a|a_0)$';
ax3.set_xlabel("Age (years)");
ax3.set_ylabel(labelString); 
ax3.legend(labelArray);   
    
ax4.grid();
ax4.set_xlabel("Age (years)");
labelString = r'$p_{a|a_0}(a|a_0)$';
ax4.set_ylabel(labelString);
ax4.legend(labelArray); 

lifeExpectancy = calcConditionalLifeExpect(cdf);
fig = plt.figure();
ax5 = fig.add_subplot(1, 1, 1);
ax5.plot(ageArrayTotal[0:-1], lifeExpectancy,'.-');
ax5.grid();
ax5.set_xlabel("Current age (years)");
ax5.set_ylabel("Expected death Age");

