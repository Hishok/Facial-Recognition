% Augment images
% Preprocessing - script 3
%% Augment Images 

% Augmentation of images only applied to original images taken and not to
% frame extractions from the videos.

directory_path = '/Users/hishorajanathan/Desktop/CV/CroppedFaces';

individual_folders = dir(directory_path);

%% Augment images 

% Resize image to scale
scale = [80 80];
original_video = 'OrigImg_'; % only augment original images not frame extraction
% Iterate over all the individual folders
for j = 3:length(individual_folders)
    individual_name = individual_folders(j).name;  %extract the number for each individual
    % extract all .jpg in the folder
    individual_folder = strcat(directory_path,'/' ,individual_name);
    crop_images = dir(fullfile(individual_folder,'*.jpg'));
    for k = 1:length(crop_images)
        % Find the name of the image
        ind_name = crop_images(k).name;
        frames_path = strfind(ind_name,original_video);
            if ~isempty(frames_path)
            % Get File path of image
            read_image = strcat(individual_folder,'/',ind_name);
            % Read in the current image
            current_image = imread(read_image);
            % Blur Image on original using gaussian filter 
            current_blur = imgaussfilt(current_image,10);
            
            current_blur2 = imgaussfilt(current_image,12);
            
            %Dilate original image using a ball
            se = offsetstrel('ball',5,3);
            current_dilated = imdilate(current_image,se);
       
            %adjust brightness 
            bright_blur = current_image + 100;
            bright_blur = imgaussfilt(bright_blur,6);
            
            dark_blur = current_image - 100;
            dark_blur = imgaussfilt(dark_blur,6);

            % Rotate Image
            Rotate_1 = imrotate(current_image,15,'crop');
            Rotate_1_blur = imgaussfilt(Rotate_1,6);           
            Rotate_2 = imrotate(current_image,30,'crop');
            Rotate_5 = imrotate(current_image,330,'crop');
            Rotate_5_bright  = Rotate_5 +50;
            Rotate_5_bright = imgaussfilt(Rotate_5_bright,8);

            % write image to folder
            %file path of folder where image gets saved
            final_folder = strcat(directory_path,'/',individual_name);
            %file path of the resized image 
            final_path = strcat(directory_path,'/',individual_name,'/','Blur_',num2str(j),'_',num2str(k),'.jpg');
            final_path2 = strcat(directory_path,'/',individual_name,'/','Blur2_',num2str(j),'_',num2str(k),'.jpg');
            final_path_rot1 = strcat(directory_path,'/',individual_name,'/','Rot1_',num2str(j),'_',num2str(k),'.jpg');
            final_path_rot1_blur = strcat(directory_path,'/',individual_name,'/','Rot1blur_',num2str(j),'_',num2str(k),'.jpg');
            final_path_rot2 = strcat(directory_path,'/',individual_name,'/','Rot2_',num2str(j),'_',num2str(k),'.jpg');
            final_path_rot5 = strcat(directory_path,'/',individual_name,'/','Rot5_',num2str(j),'_',num2str(k),'.jpg');
            final_path_rot5bright = strcat(directory_path,'/',individual_name,'/','Rot5bright_',num2str(j),'_',num2str(k),'.jpg');
            
            final_path_brightblur = strcat(directory_path,'/',individual_name,'/','BrightBlur_',num2str(j),'_',num2str(k),'.jpg');
            final_path_darkblur = strcat(directory_path,'/',individual_name,'/','DarkBlur_',num2str(j),'_',num2str(k),'.jpg');
            final_path_dilated = strcat(directory_path,'/',individual_name,'/','dilated_',num2str(j),'_',num2str(k),'.jpg');
            
            % if the folder does not exist, create a new folder to write
            % the image
            if ~exist(final_folder,'dir')
                mkdir(final_folder);
            end
            %write cropped face image to the folder specified
            imwrite(current_blur,final_path)
            imwrite(current_blur2,final_path2)
            imwrite(Rotate_1,final_path_rot1)
            imwrite(Rotate_1_blur,final_path_rot1_blur)
            imwrite(Rotate_2,final_path_rot2)
            imwrite(Rotate_5,final_path_rot5)
            imwrite(Rotate_5_bright,final_path_rot5bright)
            imwrite(bright_blur,final_path_brightblur)
            imwrite(dark_blur,final_path_darkblur)
            imwrite(current_dilated,final_path_dilated)
            end
    end
end

%% Find out the number of images per folder 
% Not all folders will have the correct number of images, we need to
% balance them. We will set a minimum of 150 images per folder. Any folder
% that does not have this limit extra copies will be made 

files_number = zeros(length(individual_folders),2);

original_video2 = 'Image_from_video_';

for j = 3:length(individual_folders)
    individual_name = individual_folders(j).name;  %extract the number for each individual 
    if individual_name == ".DS_Store"
        continue
    end
    % extract all .jpg in the folder
    individual_folder = strcat(directory_path,'/' ,individual_name);
    crop_images = dir(fullfile(individual_folder,'*.jpg'));
    if length(crop_images) < 200 
        %find number of files that need to be added 
        extra_files = 200 - length(crop_images);
        % copy random files
        %randomnum = randperm(length(crop_images),extra_files);
        for i  = 1:extra_files
           %choose random file in folder 
           randomfileindex = randi(numel(crop_images));
           randomfile = crop_images(randomfileindex).name;
           randomfile2 = strcat(individual_folder,'/',randomfile);
           % To extract the name of the file excluding extension
           [pathstr,name,ext] = fileparts(randomfile);
           writepath = strcat(individual_folder,'/',name,'_copy_',num2str(i),'.jpg');
           copyfile(randomfile2,writepath);
        end
    end
end
   
%% Resize all images 

% Resize all the images to 80 x 80 for the SVM and MLP models.

write_path2 = '/Users/hishorajanathan/Desktop/CV/CroppedFacesAugment';

for j = 3:length(individual_folders)
    individual_name = individual_folders(j).name;  %extract the number for each individual
    % extract all .jpg in the folder
    individual_folder = strcat(directory_path,'/' ,individual_name);
    crop_images = dir(fullfile(individual_folder,'*.jpg'));
    for k = 1:length(crop_images)
        % Find the name of the image
        ind_name = crop_images(k).name;
        read_image = strcat(individual_folder,'/',ind_name);
        % Read in the current image
        current_image = imread(read_image);
        ind_name_resize = imresize(current_image,scale);
        % if the folder does not exist, create a new folder to write
        % the image
        final_folder = strcat(write_path2,'/',individual_name);
        final_path_size = strcat(final_folder,'/',ind_name);
        if ~exist(final_folder,'dir')
            mkdir(final_folder);
        end
        imwrite(ind_name_resize,final_path_size);
    end
end
        