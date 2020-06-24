INM460 - Computer Vision Submission
Hisho Rajanathan
LINK TO DROPBOX - https://www.dropbox.com/sh/fgrradxel8wfdg7/AAD0d4m1P_KOf3tMtEpjPxf5a?dl=0
#################################################################################
NOTE: The files have been split and all the models are available on DropBox
LINK TO DROPBOX - https://www.dropbox.com/sh/fgrradxel8wfdg7/AAD0d4m1P_KOf3tMtEpjPxf5a?dl=0
#################################################################################

.ZIP1 -
Files included: 
1. framextraction.m
2. detectFace.m
3. augmentImages.m
4. trainClassifer.m
5. SURF_HOG.m
6. RecogniseFace.m
7. FinalScreenRecording.mp4 - to show how the recognise face function working

###########################################################################
FILES AVAILABLE ON DROPBOX  - LINK TO DROPBOX - https://www.dropbox.com/sh/fgrradxel8wfdg7/AAD0d4m1P_KOf3tMtEpjPxf5a?dl=0
############################################################################
SAVE MODELS AND FACE MASK IMAGE IN SAME FOLDER AS RecogniseFace.m for the function to work correctly. Thank you !

The files required for RecogniseFace.m function are also saved in the folder
1. CNN_AlexNet.mat
2. SVM_SURF.mat
3. SVM_HOG.mat
4. MLP_HOG.mat
5. MLP_SURF.mat
6. SURF_bagFeatures.mat
7. Guy_Fawkes_Mask.PNG

The files above are required to successfully use the RecogniseFace function. Number 7 loads the face mask for the creative section.

The RecogniseFace function returns a P matrix which describes the student(s) in an RGB image.

P = RecogniseFace(I,featureType,classifierType,creativeMode)

I - specify the file path of the image
Feature Type - “HOG”,”SURF”,””
Classifier Type - “MLP,”SVM”,”CNN”
Creative Mode - 1 on, 0 off. DO NOT PUT "" AROUND THE VALUES for creative mode

Example
RecogniseFace('IMG_6844.JPG','HOG','MLP',0)

Creative Mode Example
RecogniseFace('IMG_7056.JPG','','CNN',1)