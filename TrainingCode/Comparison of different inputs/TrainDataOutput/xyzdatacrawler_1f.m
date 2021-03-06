addpath('/home/student/recent/lib');
addpath('/home/student/recent/NDLUTIL0p161/NDLUTIL0p161');
format long;
folders=dir('/home/student/recent/data_full');
train_x=fopen('/home/student/DataPrepare/Comparison of different inputs/TrainDataOutput/train_x.txt','a');
train_y=fopen('/home/student/DataPrepare/Comparison of different inputs/TrainDataOutput/train_y.txt','a');
test_x=fopen('/home/student/DataPrepare/Comparison of different inputs/TrainDataOutput/test_x.txt','a');
test_y=fopen('/home/student/DataPrepare/Comparison of different inputs/TrainDataOutput/test_y.txt','a');
%for i=3:length(folders)
for i=3:12
    folder=strcat('/home/student/recent/data_full/',folders(i).name);
    files=dir(folder);
    for j=3:floor(length(files)/10*9)
        file_add=strcat(folder,'/',files(j).name)
        [skel,channels,frameLength] = bvhReadFile(file_add);
        %calculate the arm
        xyz_temp=bvh2xyz(skel,channels(1,:));
        length_arm=norm(xyz_temp(33,:)-xyz_temp(32,:))+norm(xyz_temp(32,:)-xyz_temp(31,:));
        channels_normal=channels/180;
        for x=1:size(channels,1)
            %normalize xyz
           xyz_full= bvh2xyz(skel, channels(x,:));
           xyz=(xyz_full(33,:)-xyz_full(31,:))./length_arm;
           fprintf(train_x,'%.15f %.15f %.15f \n',...
               xyz(1),xyz(2),xyz(3));
           fprintf(train_y,'%.15f %.15f %.15f %.15f %.15f %.15f\n',...
               channels_normal(x,79),channels_normal(x,80),...
               channels_normal(x,81),channels_normal(x,82),channels_normal(x,83),channels_normal(x,84));
        end
    end
    for j=(floor(length(files)/10*9)+1):length(files)
        file_add=strcat(folder,'/',files(j).name)
        [skel,channels,frameLength] = bvhReadFile(file_add);
        %calculate the arm
        xyz_temp=bvh2xyz(skel,channels(1,:));
        length_arm=norm(xyz_temp(33,:)-xyz_temp(32,:))+norm(xyz_temp(32,:)-xyz_temp(31,:));
        channels_normal=channels/180;
        for x=1:size(channels,1)
            %normalize xyz
           xyz_full= bvh2xyz(skel, channels(x,:));
           xyz=(xyz_full(33,:)-xyz_full(31,:))./length_arm;
           fprintf(test_x,'%.15f %.15f %.15f \n',...
               xyz(1),xyz(2),xyz(3));
           fprintf(test_y,'%.15f %.15f %.15f %.15f %.15f %.15f\n',...
               channels_normal(x,79),channels_normal(x,80),...
               channels_normal(x,81),channels_normal(x,82),channels_normal(x,83),channels_normal(x,84));
        end
    end
end
fclose(train_x);
fclose(train_y);
fclose(test_x);
fclose(test_y);