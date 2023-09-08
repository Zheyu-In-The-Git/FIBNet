import numpy as np
import pandas



# CelebA, LFW, Adience

arcface_eer = np.array([6.625,	4.133,	18.16])
IBNets_00001_eer = np.array([9.893,	8.566,	21.89])
IBNets_0001_eer = np.array([11.73,	10.16,	19.06])
IBNets_001_eer = np.array([19.07,	20.93,	31.76])
IBNets_01_eer = np.array([23.86, 27.06,	28.62])
IBNets_1_eer = np.array([50.22, 47.73, 47.91])



IBNets_00001_eer_rate = (IBNets_00001_eer - arcface_eer) / arcface_eer
IBNets_0001_eer_rate = (IBNets_0001_eer - arcface_eer) / arcface_eer
IBNets_001_eer_rate = (IBNets_001_eer - arcface_eer) / arcface_eer
IBNets_01_eer_rate = (IBNets_01_eer - arcface_eer) / arcface_eer
IBNets_1_eer_rate = (IBNets_1_eer - arcface_eer) / arcface_eer






'''
#####################################
### LR gender_fic ###################
#####################################
'''


# CelebA, LFW, Adience
arcface_LR_gender_acc = np.array([ 74.72, 53.26, 59.69])
IBNets_00001_LR_gender_acc = np.array([ 78.83, 55.31,	63.76])
IBNets_0001_LR_gender_acc = np.array([ 76.92,	51.25,	62.57])
IBNets_001_LR_gender_acc = np.array([ 73.33, 49.78, 59.73])
IBNets_01_LR_gender_acc = np.array([ 59.25, 34.4, 50.5])
IBNets_1_LR_gender_acc = np.array([ 58.85, 22.54, 50.41])



arcface_LR_gender_fic = 100 - arcface_LR_gender_acc
IBNets_00001_LR_gender_fic = 100 -IBNets_00001_LR_gender_acc
IBNets_0001_LR_gender_fic = 100 -IBNets_0001_LR_gender_acc
IBNets_001_LR_gender_fic = 100 -IBNets_001_LR_gender_acc
IBNets_01_LR_gender_fic = 100 -IBNets_01_LR_gender_acc
IBNets_1_LR_gender_fic = 100 -IBNets_1_LR_gender_acc



#print(arcface_LR_gender_fic)
#print(IBNets_00001_LR_gender_fic)
#print(IBNets_00001_LR_gender_fic-arcface_LR_gender_fic)

#print('!!!!')
#print((IBNets_00001_LR_gender_fic - arcface_LR_gender_fic ) / arcface_LR_gender_fic)

IBNets_00001_LR_gender_fic_rate = (IBNets_00001_LR_gender_fic - arcface_LR_gender_fic ) / arcface_LR_gender_fic
IBNets_0001_LR_gender_fic_rate = (IBNets_0001_LR_gender_fic - arcface_LR_gender_fic ) / arcface_LR_gender_fic
IBNets_001_LR_gender_fic_rate = (IBNets_001_LR_gender_fic - arcface_LR_gender_fic ) / arcface_LR_gender_fic
IBNets_01_LR_gender_fic_rate = (IBNets_01_LR_gender_fic - arcface_LR_gender_fic ) / arcface_LR_gender_fic
IBNets_1_LR_gender_fic_rate = (IBNets_1_LR_gender_fic - arcface_LR_gender_fic ) / arcface_LR_gender_fic


IBNets_00001_LR_gender_PIC = IBNets_00001_LR_gender_fic_rate - IBNets_00001_eer_rate
IBNets_0001_LR_gender_PIC = IBNets_0001_LR_gender_fic_rate - IBNets_0001_eer_rate
IBNets_001_LR_gender_PIC = IBNets_001_LR_gender_fic_rate - IBNets_001_eer_rate
IBNets_01_LR_gender_PIC = IBNets_01_LR_gender_fic_rate - IBNets_01_eer_rate
IBNets_1_LR_gender_PIC = IBNets_1_LR_gender_fic_rate - IBNets_1_eer_rate
#print('IBNets_00001_LR_gender_PIC', IBNets_00001_LR_gender_PIC)
#print('IBNets_0001_LR_gender_PIC', IBNets_0001_LR_gender_PIC)
#print('IBNets_001_LR_gender_PIC', IBNets_001_LR_gender_PIC)
#print('IBNets_01_LR_gender_PIC', IBNets_01_LR_gender_PIC)
#print('IBNets_1_LR_gender_PIC', IBNets_1_LR_gender_PIC)




'''
#####################################
### MLP gender_fic ###################
#####################################
'''


arcface_MLP_gender_acc = np.array([ 97.7, 91.38, 71.93])
IBNets_00001_MLP_gender_acc = np.array([ 95.01,	86.15,	68.23])
IBNets_0001_MLP_gender_acc = np.array([ 94.89,	86.02,	68.61])
IBNets_001_MLP_gender_acc = np.array([87.82,	75.42,	62.8 ])
IBNets_01_MLP_gender_acc = np.array([85.96,	72.81,	60.26 ])
IBNets_1_MLP_gender_acc = np.array([ 58.85,	22.54,	50.41])


arcface_MLP_gender_fic = 100 - arcface_MLP_gender_acc
IBNets_00001_MLP_gender_fic = 100 -IBNets_00001_MLP_gender_acc
IBNets_0001_MLP_gender_fic = 100 -IBNets_0001_MLP_gender_acc
IBNets_001_MLP_gender_fic = 100 -IBNets_001_MLP_gender_acc
IBNets_01_MLP_gender_fic = 100 -IBNets_01_MLP_gender_acc
IBNets_1_MLP_gender_fic = 100 -IBNets_1_MLP_gender_acc


IBNets_00001_MLP_gender_fic_rate = (IBNets_00001_MLP_gender_fic - arcface_MLP_gender_fic ) / arcface_MLP_gender_fic
IBNets_0001_MLP_gender_fic_rate = (IBNets_0001_MLP_gender_fic - arcface_MLP_gender_fic ) / arcface_MLP_gender_fic
IBNets_001_MLP_gender_fic_rate = (IBNets_001_MLP_gender_fic - arcface_MLP_gender_fic ) / arcface_MLP_gender_fic
IBNets_01_MLP_gender_fic_rate = (IBNets_01_MLP_gender_fic - arcface_MLP_gender_fic ) / arcface_MLP_gender_fic
IBNets_1_MLP_gender_fic_rate = (IBNets_1_MLP_gender_fic - arcface_MLP_gender_fic ) / arcface_MLP_gender_fic


IBNets_00001_MLP_gender_PIC = IBNets_00001_MLP_gender_fic_rate - IBNets_00001_eer_rate
IBNets_0001_MLP_gender_PIC = IBNets_0001_MLP_gender_fic_rate - IBNets_0001_eer_rate
IBNets_001_MLP_gender_PIC = IBNets_001_MLP_gender_fic_rate - IBNets_001_eer_rate
IBNets_01_MLP_gender_PIC = IBNets_01_MLP_gender_fic_rate - IBNets_01_eer_rate
IBNets_1_MLP_gender_PIC = IBNets_1_MLP_gender_fic_rate - IBNets_1_eer_rate



#print('IBNets_00001_MLP_gender_PIC', IBNets_00001_MLP_gender_PIC)
#print('IBNets_0001_MLP_gender_PIC', IBNets_0001_MLP_gender_PIC)
#print('IBNets_001_MLP_gender_PIC', IBNets_001_MLP_gender_PIC)
#print('IBNets_01_MLP_gender_PIC', IBNets_01_MLP_gender_PIC)
#print('IBNets_1_MLP_gender_PIC', IBNets_1_MLP_gender_PIC)



'''
#####################################
### LR race_fic ###################
#####################################
'''

# CelebA, LFW, Adience
arcface_LR_race_acc = np.array([ 72.91,	74.18,	53.7])
IBNets_00001_LR_race_acc = np.array([ 74.33, 74.12,	56.01])
IBNets_0001_LR_race_acc = np.array([ 73.4,	75.13,	54.27])
IBNets_001_LR_race_acc = np.array([ 71.86,	73.29,	52.9])
IBNets_01_LR_race_acc = np.array([ 68.02,	74.07,	48.53])
IBNets_1_LR_race_acc = np.array([ 68.54,	74.79,	48.53])



arcface_LR_race_fic = 100 - arcface_LR_race_acc
IBNets_00001_LR_race_fic = 100 -IBNets_00001_LR_race_acc
IBNets_0001_LR_race_fic = 100 -IBNets_0001_LR_race_acc
IBNets_001_LR_race_fic = 100 -IBNets_001_LR_race_acc
IBNets_01_LR_race_fic = 100 -IBNets_01_LR_race_acc
IBNets_1_LR_race_fic = 100 -IBNets_1_LR_race_acc



#print(arcface_LR_gender_fic)
#print(IBNets_00001_LR_gender_fic)
#print(IBNets_00001_LR_gender_fic-arcface_LR_gender_fic)

#print('!!!!')
#print((IBNets_00001_LR_gender_fic - arcface_LR_gender_fic ) / arcface_LR_gender_fic)

IBNets_00001_LR_race_fic_rate = (IBNets_00001_LR_race_fic - arcface_LR_race_fic ) / arcface_LR_race_fic
IBNets_0001_LR_race_fic_rate = (IBNets_0001_LR_race_fic - arcface_LR_race_fic ) / arcface_LR_race_fic
IBNets_001_LR_race_fic_rate = (IBNets_001_LR_race_fic - arcface_LR_race_fic ) / arcface_LR_race_fic
IBNets_01_LR_race_fic_rate = (IBNets_01_LR_race_fic - arcface_LR_race_fic ) / arcface_LR_race_fic
IBNets_1_LR_race_fic_rate = (IBNets_1_LR_race_fic - arcface_LR_race_fic ) / arcface_LR_race_fic


IBNets_00001_LR_race_PIC = IBNets_00001_LR_race_fic_rate - IBNets_00001_eer_rate
IBNets_0001_LR_race_PIC = IBNets_0001_LR_race_fic_rate - IBNets_0001_eer_rate
IBNets_001_LR_race_PIC = IBNets_001_LR_race_fic_rate - IBNets_001_eer_rate
IBNets_01_LR_race_PIC = IBNets_01_LR_race_fic_rate - IBNets_01_eer_rate
IBNets_1_LR_race_PIC = IBNets_1_LR_race_fic_rate - IBNets_1_eer_rate
#print('IBNets_00001_LR_race_PIC', IBNets_00001_LR_race_PIC)
#print('IBNets_0001_LR_race_PIC', IBNets_0001_LR_race_PIC)
#print('IBNets_001_LR_race_PIC', IBNets_001_LR_race_PIC)
#print('IBNets_01_LR_race_PIC', IBNets_01_LR_race_PIC)
#print('IBNets_1_LR_race_PIC', IBNets_1_LR_race_PIC)




'''
#####################################
### MLP race_fic ###################
#####################################
'''


# CelebA, LFW, Adience
arcface_MLP_race_acc = np.array([ 79.53,	77.72,	62.71])
IBNets_00001_MLP_race_acc = np.array([ 80.44,	77.67,	60.63])
IBNets_0001_MLP_race_acc = np.array([ 80.46,	76.98,	61.5])
IBNets_001_MLP_race_acc = np.array([ 78.37,	77.58,	56.3])
IBNets_01_MLP_race_acc = np.array([ 76.77,	76.97,	57.3])
IBNets_1_MLP_race_acc = np.array([ 68.54,	74.79,	48.53])



arcface_MLP_race_fic = 100 - arcface_MLP_race_acc
IBNets_00001_MLP_race_fic = 100 -IBNets_00001_MLP_race_acc
IBNets_0001_MLP_race_fic = 100 -IBNets_0001_MLP_race_acc
IBNets_001_MLP_race_fic = 100 -IBNets_001_MLP_race_acc
IBNets_01_MLP_race_fic = 100 -IBNets_01_MLP_race_acc
IBNets_1_MLP_race_fic = 100 -IBNets_1_MLP_race_acc



#print(arcface_LR_gender_fic)
#print(IBNets_00001_LR_gender_fic)
#print(IBNets_00001_LR_gender_fic-arcface_LR_gender_fic)

#print('!!!!')
#print((IBNets_00001_LR_gender_fic - arcface_LR_gender_fic ) / arcface_LR_gender_fic)

IBNets_00001_MLP_race_fic_rate = (IBNets_00001_MLP_race_fic - arcface_MLP_race_fic ) / arcface_MLP_race_fic
IBNets_0001_MLP_race_fic_rate = (IBNets_0001_MLP_race_fic - arcface_MLP_race_fic ) / arcface_MLP_race_fic
IBNets_001_MLP_race_fic_rate = (IBNets_001_MLP_race_fic - arcface_MLP_race_fic ) / arcface_MLP_race_fic
IBNets_01_MLP_race_fic_rate = (IBNets_01_MLP_race_fic - arcface_MLP_race_fic ) / arcface_MLP_race_fic
IBNets_1_MLP_race_fic_rate = (IBNets_1_MLP_race_fic - arcface_MLP_race_fic ) / arcface_MLP_race_fic


IBNets_00001_MLP_race_PIC = IBNets_00001_MLP_race_fic_rate - IBNets_00001_eer_rate
IBNets_0001_MLP_race_PIC = IBNets_0001_MLP_race_fic_rate - IBNets_0001_eer_rate
IBNets_001_MLP_race_PIC = IBNets_001_MLP_race_fic_rate - IBNets_001_eer_rate
IBNets_01_MLP_race_PIC = IBNets_01_MLP_race_fic_rate - IBNets_01_eer_rate
IBNets_1_MLP_race_PIC = IBNets_1_MLP_race_fic_rate - IBNets_1_eer_rate
#print('IBNets_00001_MLP_race_PIC', IBNets_00001_MLP_race_PIC)
#print('IBNets_0001_MLP_race_PIC', IBNets_0001_MLP_race_PIC)
#print('IBNets_001_MLP_race_PIC', IBNets_001_MLP_race_PIC)
#print('IBNets_01_MLP_race_PIC', IBNets_01_MLP_race_PIC)
#print('IBNets_1_MLP_race_PIC', IBNets_1_MLP_race_PIC)



'''
########################################
### PFRNet_gender_fic ##################
########################################
'''
PFRNet_eer = np.array([13.0142,	13.1333, 31.6734])

PFRNet_LR_gender_acc = np.array([71.18, 57.06, 58.43])
PFRNet_LR_gender_fic = 100-PFRNet_LR_gender_acc
PFRNet_LR_gender_fic_rate = (PFRNet_LR_gender_fic - arcface_LR_gender_fic)/ arcface_LR_gender_fic
PFRNet_LR_eer_rate = (PFRNet_eer - arcface_eer) / arcface_eer
PFRNet_LR_gender_pic = PFRNet_LR_gender_fic_rate - PFRNet_LR_eer_rate
#print(PFRNet_LR_gender_pic)


PFRNet_MLP_gender_acc = np.array([93.96, 86.11, 67.66])
PFRNet_MLP_gender_fic = 100 - PFRNet_MLP_gender_acc
PFRNet_MLP_gender_fic_rate = (PFRNet_MLP_gender_fic - arcface_MLP_gender_fic)/arcface_MLP_gender_fic
PFRNet_MLP_eer_rate = (PFRNet_eer - arcface_eer) / arcface_eer
PFRNet_MLP_gender_pic = PFRNet_MLP_gender_fic_rate - PFRNet_MLP_eer_rate
#print(PFRNet_MLP_gender_pic)

'''
########################################
### PFRNet_race_fic ##################
########################################
'''
PFRNet_LR_race_acc = np.array([70.69, 73.29, 50.78])
PFRNet_LR_race_fic = 100 - PFRNet_LR_race_acc
PFRNet_LR_race_fic_rate = (PFRNet_LR_race_fic - arcface_LR_race_fic) / arcface_LR_race_fic
PFRNet_LR_eer_rate = (PFRNet_eer - arcface_eer)/arcface_eer
PFRNet_LR_race_pic = PFRNet_LR_race_fic_rate - PFRNet_LR_eer_rate
#print(PFRNet_LR_race_pic)


PFRNet_MLP_race_acc = np.array([76.67, 70.39, 59.53])
PFRNet_MLP_race_fic = 100 - PFRNet_MLP_race_acc
PFRNet_MLP_race_fic_rate = (PFRNet_MLP_race_fic - arcface_MLP_race_fic) / arcface_MLP_race_fic
PFRNet_MLP_eer_rate = (PFRNet_eer - arcface_eer)/arcface_eer
PFRNet_MLP_race_pic = PFRNet_MLP_race_fic_rate - PFRNet_MLP_eer_rate
# print(PFRNet_MLP_race_pic)

'''
#####################################
### RAPP_gender_LR ##################
#####################################
'''
RAPP_eer = np.array([30.819, 21.866, 27.63])

RAPP_LR_gender_acc = np.array([77.14, 54.89, 57.63])
RAPP_LR_gender_fic = 100 - RAPP_LR_gender_acc
RAPP_LR_gender_fic_rate = (RAPP_LR_gender_fic - arcface_LR_gender_fic) / arcface_LR_gender_fic
RAPP_LR_eer_rate = (RAPP_eer - arcface_eer)/arcface_eer
RAPP_LR_gender_pic = RAPP_LR_gender_fic_rate - RAPP_LR_eer_rate
#print(RAPP_LR_gender_pic)


RAPP_MLP_gender_acc = np.array([96.21, 89.26, 65.85])
RAPP_MLP_gender_fic = 100 - RAPP_MLP_gender_acc
RAPP_MLP_gender_fic_rate = (RAPP_MLP_gender_fic - arcface_MLP_gender_fic) / arcface_MLP_gender_fic
RAPP_MLP_gender_eer_rate = (RAPP_eer - arcface_eer) / arcface_eer
RAPP_MLP_gender_pic = RAPP_MLP_gender_fic_rate - RAPP_MLP_gender_eer_rate
#print(RAPP_MLP_gender_pic)

RAPP_LR_race_acc = np.array([72.43, 72.06, 53.78])
RAPP_LR_race_fic = 100-RAPP_LR_race_acc
RAPP_LR_race_fic_rate = (RAPP_LR_race_fic - arcface_LR_race_fic) / arcface_LR_race_fic
RAPP_LR_race_eer_rate = (RAPP_eer-arcface_eer)/arcface_eer
RAPP_LR_race_pic = RAPP_LR_race_fic_rate - RAPP_LR_race_eer_rate
#print(RAPP_LR_race_pic)

RAPP_MLP_race_acc = np.array([81.26, 75.9, 60.87])
RAPP_MLP_race_fic = 100 - RAPP_MLP_race_acc
RAPP_MLP_race_fic_rate = (RAPP_MLP_race_fic - arcface_MLP_race_fic) / arcface_MLP_race_fic
RAPP_MLP_race_eer_rate = (RAPP_eer - arcface_eer) / arcface_eer
RAPP_MLP_race_pic = RAPP_MLP_race_fic_rate - RAPP_MLP_race_eer_rate
print(RAPP_MLP_race_pic)















