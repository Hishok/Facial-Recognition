%% Frame Extraction
% This script is used to read in the short videos with .mp4 extension and
% extract an x number of frames from the video.
% Preprocessing - script 1 

%% Directory of images

% Define the file path for where individual images are stored 
directory_path = '/Users/hishorajanathan/Desktop/CV/Individual';
folder = dir(directory_path);
%% Extract frames from video

num_frames = 30; % number of frames to extract

for i = 3:size(folder,1)
    %find the folder name
    folder_name = folder(i).name; %find the label of each person
    folder_dir = strcat(directory_path,'/' ,folder_name);
    folder_files = dir(folder_dir);
    %only read in .mp4 files extension
    videos = dir(fullfile(folder_dir,'*.mp4'));
    
    for j = 1:size(videos,1)
        a = VideoReader(strcat(directory_path,'/',folder_name,'/',videos(j).name)); 
        % if a video cannot be read skip to the next iteration
        try
            images = read(a);
        catch
            continue
        end
        images = read(a);
        for k = 1:num_frames
            % if a frame cannot be extracted from the video skip to the
            % next iteration
            try
                I = images(:,:,:,k);
            catch
                continue
            end
            I = images(:,:,:,k);
            %convert to grayscale to include brightness threshold
            grayI = rgb2gray(I);
            brightnessI = mean(mean(I));
            %brightness threshold of 64 to avoid saving frames which are
            %blacked out
            if brightnessI > 64
                %write the file to folder with a predefined file name
                output_file = strcat(directory_path,'/',folder_name,'/','Image_from_video_',num2str(j),'_frame_',num2str(k),'.jpg');
                imwrite(I,output_file);
            end
        end
    end
end
