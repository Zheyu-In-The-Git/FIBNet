import numpy as np
import pandas



# CelebA, LFW_CASIA, Adience

arcface_eer = np.array([6.625,	4.714,	18.16])
IBNets_00001_eer = np.array([9.893,	8.5428,	21.89])
IBNets_0001_eer = np.array([11.73,	10.6,	19.06])
IBNets_001_eer = np.array([19.07,	20.68,	31.76])
IBNets_01_eer = np.array([23.86, 27.37,	28.62])
IBNets_1_eer = np.array([50.22, 50.62, 47.91])



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


# CelebA, LFW_CASIA, Adience
arcface_LR_gender_acc = np.array([ 74.72, 51.22, 59.69])
IBNets_00001_LR_gender_acc = np.array([ 78.83, 53.03,	63.76])
IBNets_0001_LR_gender_acc = np.array([ 76.92,	49.45,	62.57])
IBNets_001_LR_gender_acc = np.array([ 73.33, 49.01, 59.73])
IBNets_01_LR_gender_acc = np.array([ 59.25, 34.85, 50.5])
IBNets_1_LR_gender_acc = np.array([ 58.85, 21.65, 50.41])



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


arcface_MLP_gender_acc = np.array([ 97.7, 87.09, 71.93])
IBNets_00001_MLP_gender_acc = np.array([ 95.01,	81.72,	68.23])
IBNets_0001_MLP_gender_acc = np.array([ 94.89,	80.92,	68.61])
IBNets_001_MLP_gender_acc = np.array([87.82,	70.89,	62.8 ])
IBNets_01_MLP_gender_acc = np.array([85.96,	68.23,	60.26 ])
IBNets_1_MLP_gender_acc = np.array([ 58.85,	21.65,	50.41])


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
arcface_LR_race_acc = np.array([ 72.91,	64.13,	53.7])
IBNets_00001_LR_race_acc = np.array([ 74.33, 64.03,	56.01])
IBNets_0001_LR_race_acc = np.array([ 73.4,	64.04,	54.27])
IBNets_001_LR_race_acc = np.array([ 71.86,	62.69,	52.9])
IBNets_01_LR_race_acc = np.array([ 68.02,	62.33,	48.53])
IBNets_1_LR_race_acc = np.array([ 68.54,	61.83,	48.53])



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
arcface_MLP_race_acc = np.array([ 79.53,	74.97,	62.71])
IBNets_00001_MLP_race_acc = np.array([ 80.44,	73.62,	60.63])
IBNets_0001_MLP_race_acc = np.array([ 80.46,	73.91,	61.5])
IBNets_001_MLP_race_acc = np.array([ 78.37,	71.43,	56.3])
IBNets_01_MLP_race_acc = np.array([ 76.77,	71.23,	57.3])
IBNets_1_MLP_race_acc = np.array([ 68.54,	62.84,	48.53])



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
print('IBNets_00001_MLP_race_PIC', IBNets_00001_MLP_race_PIC)
print('IBNets_0001_MLP_race_PIC', IBNets_0001_MLP_race_PIC)
print('IBNets_001_MLP_race_PIC', IBNets_001_MLP_race_PIC)
print('IBNets_01_MLP_race_PIC', IBNets_01_MLP_race_PIC)
print('IBNets_1_MLP_race_PIC', IBNets_1_MLP_race_PIC)



'''
########################################
### PFRNet_gender_fic ##################
########################################
'''
PFRNet_eer = np.array([13.0142,	13.0286, 31.6734])

PFRNet_LR_gender_acc = np.array([71.18, 54.62, 58.43])
PFRNet_LR_gender_fic = 100-PFRNet_LR_gender_acc
PFRNet_LR_gender_fic_rate = (PFRNet_LR_gender_fic - arcface_LR_gender_fic)/ arcface_LR_gender_fic
PFRNet_LR_eer_rate = (PFRNet_eer - arcface_eer) / arcface_eer
PFRNet_LR_gender_pic = PFRNet_LR_gender_fic_rate - PFRNet_LR_eer_rate
#print('PFRNet LR gender', PFRNet_LR_gender_pic)


PFRNet_MLP_gender_acc = np.array([93.96, 80.48, 67.66])
PFRNet_MLP_gender_fic = 100 - PFRNet_MLP_gender_acc
PFRNet_MLP_gender_fic_rate = (PFRNet_MLP_gender_fic - arcface_MLP_gender_fic)/arcface_MLP_gender_fic
PFRNet_MLP_eer_rate = (PFRNet_eer - arcface_eer) / arcface_eer
PFRNet_MLP_gender_pic = PFRNet_MLP_gender_fic_rate - PFRNet_MLP_eer_rate
#print('PFRNet MLP gender', PFRNet_MLP_gender_pic)

'''
########################################
### PFRNet_race_fic ##################
########################################
'''
PFRNet_LR_race_acc = np.array([70.69, 62.88, 50.78])
PFRNet_LR_race_fic = 100 - PFRNet_LR_race_acc
PFRNet_LR_race_fic_rate = (PFRNet_LR_race_fic - arcface_LR_race_fic) / arcface_LR_race_fic
PFRNet_LR_eer_rate = (PFRNet_eer - arcface_eer)/arcface_eer
PFRNet_LR_race_pic = PFRNet_LR_race_fic_rate - PFRNet_LR_eer_rate
#print('PFRNet_LR_race_pic', PFRNet_LR_race_pic)


PFRNet_MLP_race_acc = np.array([76.67, 69.21, 59.53])
PFRNet_MLP_race_fic = 100 - PFRNet_MLP_race_acc
PFRNet_MLP_race_fic_rate = (PFRNet_MLP_race_fic - arcface_MLP_race_fic) / arcface_MLP_race_fic
PFRNet_MLP_eer_rate = (PFRNet_eer - arcface_eer)/arcface_eer
PFRNet_MLP_race_pic = PFRNet_MLP_race_fic_rate - PFRNet_MLP_eer_rate
#print('PFRNet_MLP_race_pic', PFRNet_MLP_race_pic)

'''
#####################################
### RAPP_gender_LR ##################
#####################################
'''
RAPP_eer = np.array([30.819, 22.9143, 27.63])

RAPP_LR_gender_acc = np.array([77.14, 51.921, 57.63])
RAPP_LR_gender_fic = 100 - RAPP_LR_gender_acc
RAPP_LR_gender_fic_rate = (RAPP_LR_gender_fic - arcface_LR_gender_fic) / arcface_LR_gender_fic
RAPP_LR_eer_rate = (RAPP_eer - arcface_eer)/arcface_eer
RAPP_LR_gender_pic = RAPP_LR_gender_fic_rate - RAPP_LR_eer_rate
print('RAPP_LR_gender_pic', RAPP_LR_gender_pic)


RAPP_MLP_gender_acc = np.array([96.21, 83.8649, 65.85])
RAPP_MLP_gender_fic = 100 - RAPP_MLP_gender_acc
RAPP_MLP_gender_fic_rate = (RAPP_MLP_gender_fic - arcface_MLP_gender_fic) / arcface_MLP_gender_fic
RAPP_MLP_gender_eer_rate = (RAPP_eer - arcface_eer) / arcface_eer
RAPP_MLP_gender_pic = RAPP_MLP_gender_fic_rate - RAPP_MLP_gender_eer_rate
print('RAPP_MLP_gender_pic', RAPP_MLP_gender_pic)

RAPP_LR_race_acc = np.array([72.43, 62.7437, 53.78])
RAPP_LR_race_fic = 100-RAPP_LR_race_acc
RAPP_LR_race_fic_rate = (RAPP_LR_race_fic - arcface_LR_race_fic) / arcface_LR_race_fic
RAPP_LR_race_eer_rate = (RAPP_eer-arcface_eer)/arcface_eer
RAPP_LR_race_pic = RAPP_LR_race_fic_rate - RAPP_LR_race_eer_rate
print('RAPP_LR_race_pic',RAPP_LR_race_pic)

RAPP_MLP_race_acc = np.array([81.26, 74.979, 60.87])
RAPP_MLP_race_fic = 100 - RAPP_MLP_race_acc
RAPP_MLP_race_fic_rate = (RAPP_MLP_race_fic - arcface_MLP_race_fic) / arcface_MLP_race_fic
RAPP_MLP_race_eer_rate = (RAPP_eer - arcface_eer) / arcface_eer
RAPP_MLP_race_pic = RAPP_MLP_race_fic_rate - RAPP_MLP_race_eer_rate
print('RAPP_MLP_race_pic', RAPP_MLP_race_pic)















