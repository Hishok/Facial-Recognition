%% Detect Faces and crop 

% Preprocessing - Script 2 

% Crop faces from individual image and save into a separate filde directory
% using the preset write_path file. 

directory_path = '/Users/hishorajanathan/Desktop/CV/Individual';
write_path = '/Users/hishorajanathan/Desktop/CV/CroppedFaces';

individual_folders = dir(directory_path);

%% Face Detection

% Detect objects using Viola Jones algorithm using the different
% classification methods
faceDetector = vision.CascadeObjectDetector('MergeThreshold',10,'MinSize',[100,100]); 
frontfaceDetector = vision.CascadeObjectDetector('ClassificationModel','FrontalFaceLBP','MergeThreshold',10,'MinSize',[100,100]);
profilefaceDetector = vision.CascadeObjectDetector('ClassificationModel','ProfileFace','MergeThreshold',10,'MinSize',[100,100]);

frame_video = 'Image_from_video_';

% Iterate over all the individual folders
for j = 3:length(individual_folders)
    individual_name = individual_folders(j).name;  %extract the number for each individual
    % extract all .jpg in the folder
    individual_folder = strcat(directory_path,'/' ,individual_name);
    ind_images = dir(fullfile(individual_folder,'*.jpg'));
    
    % CODE TO FIX ORIENTATION OF IMAGE TAKEN FROM: https://uk.mathworks.com/matlabcentral/answers/260607-how-to-load-a-jpg-properly
 
    for i = 1:length(ind_images)
        % Find the name of the image
        ind_name = ind_images(i).name;
        % Get File path of image
        read_image = strcat(individual_folder,'/',ind_name);
        % Read in the current image
        current_image = imread(read_image);
        % Fix orientation of image
        info = imfinfo(read_image);
        if isfield(info,'Orientation')
           orient = info(1).Orientation;
           switch orient
             case 1
                %normal, leave the data alone
             case 2
                current_image = current_image(:,end:-1:1,:);         %right to left
             case 3
                current_image = current_image(end:-1:1,end:-1:1,:);  %180 degree rotation
             case 4
                current_image = current_image(end:-1:1,:,:);         %bottom to top
             case 5
                current_image = permute(current_image, [2 1 3]);     %counterclockwise and upside down
             case 6
                current_image = rot90(current_image,3);              %undo 90 degree by rotating 270
             case 7
                current_image = rot90(current_image(end:-1:1,:,:));  %undo counterclockwise and left/right
             case 8
                current_image = rot90(current_image);                %undo 270 rotation by rotating 90
             otherwise
                warning(sprintf('unknown orientation %g ignored\n', orient));
           end
        end
        % use face detector to get box around face
        bbox = step(faceDetector,current_image);
        fbox = step(frontfaceDetector,current_image);
        pbox = step(profilefaceDetector,current_image);
        %file path of folder where image gets saved
        final_folder = strcat(write_path,'/',individual_name);
        %file path of the resized image for frame extraction 
        final_path = strcat(write_path,'/',individual_name,'/','FrameImg_',num2str(i),'.jpg');
        %file path for original image
        final_path_orig = strcat(write_path,'/',individual_name,'/','OrigImg_',num2str(i),'.jpg');
        % if the folder does not exist, create a new folder to write
        % the image
        if ~exist(final_folder,'dir')
            mkdir(final_folder);
        end
        % if bbox is not empty using Frontal Face CART execute this loop
        if ~isempty(bbox)
            %crop the image to just the face
            ind_face = imcrop(current_image,bbox(1,:)); %gets the first box
            %change file name according to origin of image - from frame
            %extraction or original image
            frames_path = strfind(ind_name,frame_video);
            if ~isempty(frames_path)
                imwrite(ind_face,final_path); 
            else
                imwrite(ind_face,final_path_orig);
            end
        % if bounding box is not empty using Frontal Face LBP 
        elseif ~isempty(fbox)
            ind_face_f = imcrop(current_image,fbox(1,:)); 
            %change file name according to origin of image - from frame
            %extraction or original image
            frames_path = strfind(ind_name,frame_video);
            if ~isempty(frames_path)
                imwrite(ind_face_f,final_path); 
            else
                imwrite(ind_face_f,final_path_orig);
            end
        % if bounding box is not empty using Profile face 
        elseif ~isempty(pbox)
            ind_face_p = imcrop(current_image,pbox(1,:));
            %change file name according to origin of image - from frame
            %extraction or original image
            frames_path = strfind(ind_name,frame_video);
            if ~isempty(frames_path)
                imwrite(ind_face_p,final_path); 
            else
                imwrite(ind_face_p,final_path_orig);
            end
        end
    end
end

       