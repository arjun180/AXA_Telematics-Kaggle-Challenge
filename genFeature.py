#Author: Arjun Chakraborty 
#AXA Telematics Challenge
#genFeature.py extracts features from raw data, and converts it into training and test set.


import csv
import numpy as np
import pandas as pd
import os
import glob

""" Reading in the x and y coordinates """
def find_points(fname):
	

	table = pd.read_csv(fname)
	df = pd.DataFrame(table)

	pointsx = []
	pointsy = []
	deltas = []
	toggle = True

	for index,row in df.iterrows():
		
		pointsx.append(row.x)
		pointsy.append(row.y)
		
	pointsx = np.array(pointsx)
	pointsy = np.array(pointsy)

	return pointsx,pointsy
	

		


""" Calculation of trip distance """
def euclidean_distance(pointsx,pointsy):
	deltas=[]
	for i in range(len(pointsx)-2):
		temp = (int(pointsx[i+2] - pointsx[i]) ** 2) + (int(pointsy[i+2] - pointsy[i]) ** 2 )
		temp = temp ** 0.5
		deltas.append(temp)
		
	s = 0
    
	for x in deltas:
		s=s+x
    
	return s,deltas

""" Calculation of total velocity and percentile velocity """
def vel_total(deltas,percentile =True):
	s=0
	t =0
	vel=[]

	for x in deltas:
		s = s + x
		t = t + 1

		vel.append(x)
	
	vel_final=np.array(vel)
	vel_final = np.mean(vel)

	# Range of percentiles
	percentile_range = [1,2,5,10,15,17,19,20]

	if percentile == True:
		

		vel_percentile=[np.percentile(vel,per*5) for per in percentile_range]
		
		vel_percentile = np.mean(vel_percentile)
		
		return vel,vel_percentile  

	
	return vel,vel_final

""" Calculation of the number of stops before a certain threshold (distance) """
def euclidean_stops(pointsx,pointsy,thresh):
	stops=[]
	count=0
	for i in range(len(pointsx)-1):
		temp = (int(pointsx[i+1] - pointsx[i]) ** 2) + (int(pointsy[i+1] - pointsy[i]) ** 2 )
		temp = temp ** 0.5

		if temp > thresh:
			stops.append(temp)
			count = count +1	
	return stops,count



""" Calculation of total acceleration and percentile of acceleration """
def totalAccel(vel,percentile=True):
	accel = []
	totalAccel = 0 
	for i in range(len(vel)-1):
		
		temp = int(vel[i] - vel[i+1])
		accel.append(temp)
		totalAccel = totalAccel + temp
	
	percentile_range = [1,2,5,10,15,17,19,20]
	

	if percentile == True:
		totalAccel_percentile=[np.percentile(totalAccel,per*5) for per in percentile_range]
		
		totalAccel_percentile = np.mean(totalAccel_percentile)
		return totalAccel_percentile

	
	return totalAccel

""" Calculation of the ratio of stops to total distance """
def ratio_stops(s,count):
	ratio = count/s
	return ratio


""" Calculation of heading angle """
def angle_car(pointsx,pointsy,percentile=True):
	
	
	angle=np.arctan(np.divide(np.diff(pointsx),np.diff(pointsy)))
	angle= angle[~np.isnan(angle)]
	angles_final = np.mean(angle)
	

	percentile_range = [1,2,5,10,15,17,19,20]
	if percentile == True:
		
		angle_percentile=[np.percentile(angles_final,per*5) for per in percentile_range]
		
		angle_percentile = np.mean(angle_percentile)
		return angle_percentile

	return angles_final

""" Calculation of count of number of stops below a certain threshold (speed)  """
def low_speedTimes(vel,thresh):
	count_low = 0
	if (vel < thresh):
		#thresh= 0.05, 0.1, 0.15, 0.2, 0.25
		count_low =  count_low  + 1

	return count_low

""" Calculation of turning aggresion and percentiles """

def turning_aggression(pointsx,pointsy,vel,percentile=False):
	turning_final = []
	angle=np.arctan(np.divide(np.diff(pointsx),np.diff(pointsy)))
	angle= angle[~np.isnan(angle)]
	turning_agg = angle*vel
	
	turning_average = np.mean(turning_agg)

	percentile_range = [1,2,5,10,15,17,19,20]

	if percentile == True:
		turning_percentile=[np.percentile(angle,per*5) for per in percentile_range]
		
		turning_percentile = np.mean(turning_percentile)
		
		return turning_percentile

	return turning_average



""" Generates all the features. Converts them into a feature matrix. Split into 70/30 Training-Test split"""
def generate_features():

	
	path = "/Users/arjun/Documents/drivers/1/*.csv"
	#path = "/Users/arjun/Documents/drivers/1/1.csv"

	# CODE DOESN'T SEEM TO BE WORKING WHEN I CALL generate_features


	fea=[]
	count = 0
	
	for i in range(1,201):

		
		
		fname = "/Users/arjun/Documents/drivers/1/%d.csv" % i

		
	
		
		pointsx,pointsy = find_points(fname)


		# Euclidean Distance
		s,deltas = euclidean_distance(pointsx,pointsy)
		
		# Normal velocity
		vel,vel_final = vel_total(deltas,percentile=False)
		

		#Percentile velocity
		vel,vel_percentile= vel_total(deltas,percentile=False)

		# Low speed times
		count_low = low_speedTimes(vel,thresh)
		

		# No of stops
		stops,counts = euclidean_stops(pointsx,pointsy,1.5)
		
		# Acceleration
		acceleration = totalAccel(vel,percentile=False)
		
		# Acceleration with percentile
		acceleration_percentile = totalAccel(vel,percentile=True)
		

		#Ratio of stops
		ratio = ratio_stops(s,counts)
		
		# Normal angle
		angle_final = angle_car(pointsx,pointsy,percentile=False)
		
		#Percentile angle
		angle_percentile = angle_car(pointsx,pointsy,percentile=True)
		
		
		# Real turning average
		turning_average = turning_aggression(pointsx,pointsy,vel_final,percentile=False)


		#Percentile turning average
		turning_percentile = turning_aggression(pointsx,pointsy,vel_final,percentile=True)
		

		fea.append([s,vel_final,vel_percentile,counts,count_low,acceleration,acceleration_percentile,ratio,angle_final,angle_percentile,turning_average,turning_percentile])

		# Conversion to numpy features array
		features_array = np.array(fea)

		i = i +1
	
	# Split into 70-30 training/test split
	training_set,test_set = train_test_split(features_array,test_size=0.3)
	
	return training_set,test_set
		




if __name__ == "__main__":

	generate_features()		
	





	
