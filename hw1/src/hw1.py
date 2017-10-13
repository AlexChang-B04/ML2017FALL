import sys
import numpy as np

attrs = ['AMB_TEMP',	'CH4',			'CO',		'NMHC',		'NO',
		 'NO2',			'NOx',			'O3',		'PM10',		'PM2.5',
		 'RAINFALL',	'RH',			'SO2',		'THC',		'WD_HR',
		 'WIND_DIREC',	'WIND_SPEED',	'WS_HR']
month = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11]
feature = [('NO', 1), ('NO2', 1), ('NO2', 2), ('NOx', 1), ('O3', 1), ('PM10', 1), ('PM10', 2),
		   ('PM2.5', 1), ('PM2.5', 2), ('RAINFALL', 1), ('SO2', 1)]
hour = 9

# Pre-process training data
raw_data = np.nan_to_num(np.genfromtxt('./data/train.csv', delimiter=',')[1:,3:]).reshape((240, 18, 24))
month_data = np.array([np.hstack(tuple([raw_data[m+d,:,:] for d in range(20)])) for m in range(0,240,20)])
feat_data = dict([(attr, np.array([month_data[m,i,:] for m in range(12)])) for i, attr in enumerate(attrs)])
matrix = np.ones(len(month)*(480-hour))
for attr, p in feature:
	matrix = np.vstack((matrix, np.array([feat_data[attr][m,i:i+hour] ** p for i in range(480-hour) for m in month]).T))
matrix = np.vstack((matrix , np.array([feat_data['PM2.5'][m,i+hour] for i in range(480-hour) for m in month])))

# Normalization
TrainSet = matrix
TrainSet_Mean = [np.mean(TrainSet[i,:]) for i in range(len(TrainSet))]
TrainSet_Std = [np.std(TrainSet[i,:]) for i in range(len(TrainSet))]
for i in range(1, len(TrainSet)-1):
	TrainSet[i,:] = (TrainSet[i,:] - TrainSet_Mean[i]) / TrainSet_Std[i]

# Initialize linear regression
w = np.zeros(TrainSet.shape[0])
Dw = np.zeros(TrainSet.shape[0])
w[-1] = -1
eta = 1
#lamda = 0.0
cnt = 0
TrainError_rms_last = 1e6

# Run linear regression
while cnt < 1e5:
	TrainError = np.transpose(np.dot(w, TrainSet))
	dw = np.transpose(np.dot(TrainSet, TrainError)) * 2 # + w * lamda * 2
	Dw = Dw + np.square(dw)
	TrainError_rms = np.mean(np.square(TrainError)) ** 0.5
	w[:-1] = w[:-1] - np.divide(eta * dw[:-1], Dw[:-1] ** 0.5)
	if abs(TrainError_rms_last - TrainError_rms) < 1e-6:
		break
	TrainError_rms_last = TrainError_rms
	cnt = cnt + 1
	#print(TrainError_rms)
#print(cnt, TrainError_rms)

# Pre-process testing data
raw_data = np.nan_to_num(np.genfromtxt(sys.argv[1], delimiter=',')[:,2:]).reshape((240,18,9))
feat_data = dict([(attr, np.array([raw_data[m,i,:] for m in range(240)])) for i, attr in enumerate(attrs)])
matrix = np.ones(240)
for attr, p in feature:
	matrix = np.vstack((matrix, np.array([feat_data[attr][i,9-hour:9] ** p for i in range(240)]).T))

# Normalization
TestSet = matrix
for i in range(1, len(TestSet)):
	TestSet[i,:] = (TestSet[i,:] - TrainSet_Mean[i]) / TrainSet_Std[i]

# Compute testing error
TestError = np.dot(w[:-1], TestSet)

print("id,value")
for i in range(240):
	print("id_" + str(i) + "," + str(TestError[i]))