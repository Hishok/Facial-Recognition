%% Train Classifier

% CNN requires images to be a size of [227 227]
% CNN can take RGB photos
% Code taken from Tutorial 9 - posted by Dr Giacomo Tarroni
% Model Script - 1
directory_path = '/Users/hishorajanathan/Desktop/CV/CroppedFaces';
imageSize = [227 227];

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

%% Load pretrained network
%load alexnet

net = alexnet;
analyzeNetwork(net)

inputSize = net.Layers(1).InputSize

%% Replace Final Layers

% Extract all layers except last 3 

layersTransfer = net.Layers(1:end-3);

numClasses = numel(categories(imagesTrain.Labels))

layers = [
    layersTransfer
    dropoutLayer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

pixelRange = [-30 30];

% Image Data Augmenter removed from code as accuracy reduced when including
% this.

% imageAugmenter = imageDataAugmenter(...
%     'RandXReflection',true,...
%     'RandYReflection',true,...
%     'RandRotation',[-30 30],...
%     'RandXTranslation',pixelRange,...
%     'RandYTranslation',pixelRange);
augimageTrain = augmentedImageDatastore(inputSize(1:2),imagesTrain);%,...
    %'DataAugmentation',imageAugmenter);

augimageValidation = augmentedImageDatastore(inputSize(1:2),imagesValidation);



%% Train the network and specify training options

options = trainingOptions('sgdm',...
    'MiniBatchSize',64,...
    'MaxEpochs',6,...
    'InitialLearnRate',0.0001,...
    'Shuffle','every-epoch',...
    'ValidationData',augimageValidation,...
    'ValidationFrequency',20,...
    'Verbose',false,...
    'ExecutionEnvironment','parallel',...
    'Plots','training-progress');

netTransfer = trainNetwork(augimageTrain,layers,options);

%% Classify Validation Images

[YPred,scores] = classify(netTransfer,augimageValidation);

%display 4 sample images with predicted labels

validx = randperm(numel(imagesValidation.Files),4);
figure
for i = 1:4 
    subplot(2,2,i)
    VI = readimage(imagesValidation,validx(i));
    imshow(VI)
    label = YPred(validx(i));
    title(string(label));
end

%% Classification accuracy on val set 
YValidation = imagesValidation.Labels;
accuracy = mean(YPred == YValidation)

%% Confusion Matrix
CNN_Matrix = confusionmat(YPred,imagesValidation.Labels)
plotconfusion(YPred,YValidation)
title('CNN Confusion Matrix')
set(findobj(gca,'type','text'),'fontsize',3)


%% Save model

save('CNN_AlexNet.mat','netTransfer');
