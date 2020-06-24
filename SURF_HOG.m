%% Train SVM 

% SVM and MLP require images to be grayscale and not RGB
% Code taken and adapted from Lab 06 - posted by Dr Giacomo Tarroni and
% from https://uk.mathworks.com/help/vision/examples/image-category-classification-using-bag-of-features.html
% Model script - 2

directory_path = '/Users/hishorajanathan/Desktop/CV/CroppedFacesAugment';

%% Create an image datastore

faceImages = imageDatastore(directory_path,'LabelSource','foldernames','IncludeSubfolders',true);

%% Balance the dataset

%Not all labels have the same amount of files so balance the number of
%images. The minimum number is 200 hence total will be 9,600 images

countLabel = countEachLabel(faceImages);
minCountImages = min(countLabel.Count);
faceImageMin = splitEachLabel(faceImages,minCountImages,'randomize');


%% Train split
% 90% for training and 10% for validation
[imagesTrain,imagesValidation] = splitEachLabel(faceImageMin,0.9,'randomize');

%% Display some of the training images

numTrainImages = numel(imagesTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imagesTrain,idx(i));
    imshow(I)
end
%% Create SURF Features
% bagofFeatures extracts SURF features from all images in all image
% categories
bag  = bagOfFeatures(imagesTrain);

%% Extract SURF 

% use encode method
trainFeatureVector = encode(bag, imagesTrain);
validationFeatureVector = encode(bag, imagesValidation);

%% Create labels 
%% Test an individual image 

% Extract hogfeatures for a single image to test. 

img_1 = readimage(imagesTrain,1000);
cellSize = [8,8];
[hog_8x8, vis8x8] = extractHOGFeatures(img_1,'CellSize',cellSize);
imshow(img_1)

plot(vis8x8)

%% SVM 

% Find the labels for the training and validation set

trainingSVM = categorical(imagesTrain.Labels);
validationSVM = categorical(imagesValidation.Labels);

%% Extract HOG Features

% Find the total number of images for the training and validation set 
numTrainImages = numel(imagesTrain.Labels);
numValidationImages = numel(imagesValidation.Labels);

% create empty matrix to store the hog features for each image 
hogFeatureSize = length(hog_8x8);
trainingFeatures = zeros(numTrainImages,hogFeatureSize,'single');
validationFeatures = zeros(numValidationImages,hogFeatureSize,'single');

% Code taken from and adapted : https://uk.mathworks.com/help/vision/examples/digit-classification-using-hog-features.html

for i = 1:numTrainImages
    img = readimage(imagesTrain, i);
    
    %convert image to grayscale to extract hog features
    img = rgb2gray(img);
    
    trainingFeatures(i, :) = extractHOGFeatures(img,'CellSize',cellSize);
end

for i = 1:numValidationImages
    img = readimage(imagesValidation, i);
    
    %convert image to grayscale to extract hog features
    img = rgb2gray(img);
    
    validationFeatures(i, :) = extractHOGFeatures(img,'CellSize',cellSize);
end


%% SURF - SVM

% Train the SVM model
SURF_SVM = fitcecoc(trainFeatureVector,trainingSVM);

% Test model on validation set 

SURF_SVM_pred = predict(SURF_SVM,validationFeatureVector);

% calculate accuracy score for SVM SURF 

accuracy_SURF_SVM = sum(SURF_SVM_pred == validationSVM ) / length(validationSVM);

%% HOG - SVM 

% Train the SVM model using HOG features 
HOG_SVM = fitcecoc(trainingFeatures,trainingSVM);

HOG_SVM_pred = predict(HOG_SVM,validationFeatures);

% accuracy SVM HOG

accuracy_HOG_SVM = sum(HOG_SVM_pred == validationSVM) / length(validationSVM);

% Apply one hot encoding to the training and validation labels
training_number = categories(trainingSVM);
validation_number = categories(validationSVM);

% create empty matrix
SURF_Training_Labels = zeros(numel(training_number),numTrainImages);
SURF_Validation_Labels = zeros(numel(validation_number),numValidationImages);

for i = 1:numTrainImages
    t_labels = trainingSVM(i);
    SURF_Training_Labels(strcmp(training_number,cellstr(t_labels)),i) =1;
end

for i = 1:numValidationImages
    t_labels = validationSVM(i);
    SURF_Validation_Labels(strcmp(validation_number,cellstr(t_labels)),i) =1;
end

%% SURF - MLP 

% train MLP network 

hiddenSizes = 80;
trainFcn = 'trainscg';

net_SURF = patternnet(hiddenSizes,trainFcn);
net_SURF = configure(net_SURF,trainFeatureVector',SURF_Training_Labels);
net_SURF = train(net_SURF,trainFeatureVector',SURF_Training_Labels); 

% predict on validation set 

SURF_MLP_pred = net_SURF(validationFeatureVector');

SURF_MLP_pred2 = [];

for i = 1:numValidationImages
    [value, number] = max(SURF_MLP_pred(:,i));
    SURF_MLP_pred2 = [SURF_MLP_pred2; validation_number(number)];
end

SURF_MLP_pred2 = categorical(str2num(cell2mat(SURF_MLP_pred2)));

% accuracy score

accuracy_SURF_MLP = sum(SURF_MLP_pred2 == validationSVM) / length(validationSVM);


%% HOG - MLP 

% train MLP network
hiddenSizesHog = 80;
trainFcnHog = 'trainscg';

% Define network architecture 
net_HOG = patternnet(hiddenSizesHog,trainFcnHog);
net_HOG = configure(net_HOG,trainingFeatures',SURF_Training_Labels);
net_HOG = train(net_HOG,trainingFeatures',SURF_Training_Labels); 

% Find the predictions for MLP network

mlp_HOG_predict = net_HOG(validationFeatures');

% create empty cell
mlp_HOG_predict2 = [];

for i = 1:numValidationImages
    [value, number] = max(mlp_HOG_predict(:,i));
    mlp_HOG_predict2 = [mlp_HOG_predict2; validation_number(number)];
end

mlp_HOG_predict2 = categorical(str2num(cell2mat(mlp_HOG_predict2)));

% accuracy score for MLP HOG 
accuracy_HOG_MLP = sum(mlp_HOG_predict2 == validationSVM) / length(validationSVM);


%% Save Models

save('SVM_SURF.mat','SURF_SVM');
save('SVM_HOG.mat','HOG_SVM');
save('MLP_HOG.mat','net_HOG');
save('MLP_SURF.mat','net_SURF');
save('SURF_bagFeatures.mat', 'bag');
