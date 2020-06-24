function P = RecogniseFace(I,featureType,classifierType,creativeMode)
% Function returns a matrix P describing the student(s) present in the RGB
% image I.

% The P matrix is a N x 3 size where N is the number of detected people in
% the image.
% The three columns of the P matrix are :
%       ID - unique number assigned to each individual
%       x - x co-ordinate of the centre of the face of the individual
%       y - y co-ordinate of the centre of the face of the individual

% The arguments required for the Recognise Face function to work correctly
% are :
%     I - file path for the image to be tested 
%     featureType- can only take the following "HOG","SURF",""
%     classifierType - can only take the following "SVM","MLP","CNN"
%     creativeMode - can only take values 1 or 0, DO NOT put "" around
%     values
% If CNN is used as classifier type, the feature type must be set to blank
% "".
% HOG and SURF features can only be used for SVM and MLP. 

% Create array to include all possible labels for each indvidual
person_labels = ["01";"02";"03";"04";"05";"06";"07";"08";"09";"10";"11";...
        "12";"13";"14";"15";"16";"17";"18";"19";"20";"21";"22";"23";"24";...
        "25";"26";"27";"28";"29";"30";"31";"32";"33";"34";"36";"38";...
        "40";"42";"44";"46";"48";"50";"52";"54";"56";"58";"60";"78"];


% Read Image I 
J = I;

% Fix Orientation of image Code taken and adapted from: https://uk.mathworks.com/matlabcentral/answers/260607-how-to-load-a-jpg-properly
I = imread(J);
info = imfinfo(J);
if isfield(info,'Orientation')
   orient = info(1).Orientation;
   switch orient
     case 1
        %normal, leave the data alone
     case 2
        I = I(:,end:-1:1,:);         %right to left
     case 3
        I = I(end:-1:1,end:-1:1,:);  %180 degree rotation
     case 4
        I = I(end:-1:1,:,:);         %bottom to top
     case 5
        I = permute(I, [2 1 3]);     %counterclockwise and upside down
     case 6
        I = rot90(I,3);              %undo 90 degree by rotating 270
     case 7
        I = rot90(I(end:-1:1,:,:));  %undo counterclockwise and left/right
     case 8
        I = rot90(I);                %undo 270 rotation by rotating 90
     otherwise
        warning(sprintf('unknown orientation %g ignored\n', orient));
   end
end

%show image 
imshow(I)

% Detect the Face - using FrontalFaceCart 
faceDetector = vision.CascadeObjectDetector('MergeThreshold',7,'MinSize',[40,40]); 

% Find bbox using the faceDetector 
bbox = step(faceDetector,I);

%number of faces detected 
number_faces = size(bbox,1);

% initialise P Matrix by making sure there are three columns
P = zeros(number_faces,3);

% Resize images for the SVM and MLP models, as this size is what was used
% to train the models
svm_mlp_scale = [80 80];

% Code below finds out what the user has input into the Recognise Face
% function and loads the models accordingly.
if strcmp(classifierType,'SVM') == 1
    if strcmp(featureType,'SURF') == 1
        load('SVM_SURF.mat','SURF_SVM');
        % load the bag of features for the SURF feature extraction
        load('SURF_bagFeatures.mat','bag');
        for i = 1:number_faces
            %crop image to just the face
            crop_image = imcrop(I,bbox(i,:));
            % resize image for the SVM model 
            resize_image = imresize(crop_image,svm_mlp_scale);
            % convert image to grayscale for SVM model 
            I_Gray = rgb2gray(resize_image);
            % Extract SURF Features
            features_SURF = encode(bag,resize_image);
            pred_label = predict(SURF_SVM,features_SURF);
            % find the correct label for the individuals detected 
            new_label = str2num(person_labels(pred_label));
            % ID of individual 
            P(i,1) = new_label;
            %get the coordinate of centre of the face
            P(i,2) = int64(round((bbox(i,1) + bbox(i,3))/2));
            P(i,3) = int64(round((bbox(i,2) + bbox(i,4))/2));
        end
    elseif strcmp(featureType,'HOG') == 1
        load('SVM_HOG.mat','HOG_SVM');
        for i = 1:number_faces
            %crop image to just the face
            crop_image = imcrop(I,bbox(i,:));
            % resize image for the SVM model 
            resize_image = imresize(crop_image,svm_mlp_scale);
            % convert image to grayscale for SVM model 
            I_Gray = rgb2gray(resize_image);
            % extract HOG features using same cell size as the training
            % model
            features_HOG = extractHOGFeatures(I_Gray,'CellSize',[8 8]);
            pred_label = predict(HOG_SVM,features_HOG);
            new_label = str2num(person_labels(pred_label));
            % ID of individual
            P(i,1) = new_label;
            %get the coordinate of centre of the face
            P(i,2) = int64(round((bbox(i,1) + bbox(i,3))/2));
            P(i,3) = int64(round((bbox(i,2) + bbox(i,4))/2));
        end
    end
elseif strcmp(classifierType,'MLP') == 1
    if strcmp(featureType,'SURF') == 1
        load('MLP_SURF.mat','net_SURF');
        load('SURF_bagFeatures.mat','bag');
        for i = 1:number_faces
            %crop image to just the face
            crop_image = imcrop(I,bbox(i,:));
            % resize image for the SVM model 
            resize_image = imresize(crop_image,svm_mlp_scale);
            % convert image to grayscale for SVM model 
            I_Gray = rgb2gray(resize_image);
            % Extract SURF features
            surf_features = encode(bag,resize_image);
            % predict the labels 
            predicted_label = net_SURF(surf_features');
            [~,predicted_label2] = max(predicted_label(:,1));
            predicted_label3 = str2num(person_labels(predicted_label2));
            P(i,1) = predicted_label3;
            %get the coordinate of centre of the face
            P(i,2) = int64(round((bbox(i,1) + bbox(i,3))/2));
            P(i,3) = int64(round((bbox(i,2) + bbox(i,4))/2));
        end
    elseif strcmp(featureType,'HOG') == 1
        load('MLP_HOG.mat','net_HOG');
        for i = 1:number_faces
            %crop image to just the face
            crop_image = imcrop(I,bbox(i,:));
            % resize image for the SVM model 
            resize_image = imresize(crop_image,svm_mlp_scale);
            % convert image to grayscale for SVM model 
            I_Gray = rgb2gray(resize_image);
            % extract HOG features
            features_HOG = extractHOGFeatures(I_Gray,'CellSize',[8 8]);
            predicted_label = net_HOG(features_HOG');
            [~,predicted_label2] = max(predicted_label(:,1));
            predicted_label3 = str2num(person_labels(predicted_label2));
            P(i,1) = predicted_label3;
            %get the coordinate of centre of the face
            P(i,2) = int64(round((bbox(i,1) + bbox(i,3))/2));
            P(i,3) = int64(round((bbox(i,2) + bbox(i,4))/2));
        end
    end
elseif strcmp(classifierType,'CNN') == 1
    if strcmp(featureType,'') == 1
        load('CNN_AlexNet.mat','netTransfer')
        for i = 1:number_faces
            crop_image = imcrop(I,bbox(i,:));
            %change the size of the image for CNN [227 227 3]
            inputSize = netTransfer.Layers(1).InputSize;
            resize_image = augmentedImageDatastore(inputSize(1:2),crop_image);
            % classify model 
            classify_image = classify(netTransfer,resize_image);
            pred_label = str2num(person_labels(classify_image));
            % fill in P Matrix
            P(i,1) = pred_label;
            %get the coordinate of centre of the face
            P(i,2) = int64(round((bbox(i,1) + bbox(i,3))/2));
            P(i,3) = int64(round((bbox(i,2) + bbox(i,4))/2));
        end
    end
else
    disp( 'ERROR: Check Inputs');
end

% Show image with bounding box with individual ID 
show_faces = insertObjectAnnotation(I,'rectangle',bbox,P(:,1),'FontSize',60,'Color','magenta');
imshow(show_faces)

% CREATIVE MODE 

if isequal(creativeMode,1)
    % read face mask image 
    mask_image = 'Guy_Fawkes_Mask.PNG';
    guy_fawkes = imread(mask_image);
    ind_faces = cell(number_faces,1);
    face_masks = cell(number_faces,1);
    photo = show_faces;
    
    for i = 1:number_faces
        face_bbox = bbox(i,:);
        % save the cropped faces and resized face mask in a separate matrix
        ind_faces{i} = imcrop(show_faces,face_bbox);
        face_masks{i} = imresize(guy_fawkes,size(ind_faces{i},1:2));
    end

    for i = 1:number_faces
        % get dimensions of each bounding box 
        x = bbox(i,1);
        y = bbox(i,2);
        w = bbox(i,3);
        h = bbox(i,4);
        
 % Code adapted from https://uk.mathworks.com/matlabcentral/answers/516880-pixeling-only-detected-face
 
        show_faces(y:y+h, x:x+w,:) = face_masks{i};
        % replace the black bacground in the face mask with the individuals
        % face
        show_faces(show_faces == 0) = photo(show_faces == 0);

        imshow(show_faces)

    end

end
% display the P matrix
disp(P)


            
            
            
        
