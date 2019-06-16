%xyzdata of tiny sample of left hand
addpath('/home/student/recent/lib');
addpath('/home/student/recent/NDLUTIL0p161/NDLUTIL0p161');
format long;
folders=dir('/home/student/recent/data_full');
train_x=fopen('/home/student/DataPrepare/Comparison of left and right side/train_x_5_10f.txt','a');
train_y=fopen('/home/student/DataPrepare/Comparison of left and right side/train_y_5_10f.txt','a');
%test_x=fopen('/home/student/tiny_sample/test_x_5_10f.txt','a');
%test_y=fopen('/home/student/tiny_sample/test_y_5_10f.txt','a');
%for i=3:length(folders)
for i=3:12
    folder=strcat('/home/student/recent/data_full/',folders(i).name);
    files=dir(folder);
    for j=3:floor(length(files)/10*9)
        file_add=strcat(folder,'/',files(j).name)
        [skel,channels,frameLength] = bvhReadFile(file_add);
        %calculate the arm
        xyz_temp=bvh2xyz(skel,channels(1,:));
        length_arm=norm(xyz_temp(24,:)-xyz_temp(23,:))+norm(xyz_temp(23,:)-xyz_temp(22,:));
        channels_normal=channels/180;
        for x=11:(size(channels,1)-10)
            %normalize xyz
           xyz_full= bvh2xyz(skel, channels(x,:));
           xyz=(xyz_full(24,:)-xyz_full(22,:))./length_arm;
           
           xyz_full_pre= bvh2xyz(skel, channels(x-10,:));
           xyz_pre=(xyz_full_pre(24,:)-xyz_full_pre(22,:))./length_arm;
           
           xyz_full_beh= bvh2xyz(skel, channels(x+10,:));
           xyz_beh=(xyz_full_beh(24,:)-xyz_full_beh(22,:))./length_arm;
           
           xyz_full_pre2= bvh2xyz(skel, channels(x-5,:));
           xyz_pre2=(xyz_full_pre2(24,:)-xyz_full_pre2(22,:))./length_arm;
           
           xyz_full_beh2= bvh2xyz(skel, channels(x+5,:));
           xyz_beh2=(xyz_full_beh2(24,:)-xyz_full_beh2(22,:))./length_arm;
           
           fprintf(train_x,'%.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f\n',...
               xyz_pre(1),xyz_pre(2),xyz_pre(3),...
               xyz_pre2(1),xyz_pre2(2),xyz_pre2(3),...
               xyz(1),xyz(2),xyz(3),...
               xyz_beh2(1),xyz_beh2(2),xyz_beh2(3),...
               xyz_beh(1),xyz_beh(2),xyz_beh(3));
           fprintf(train_y,'%.15f %.15f %.15f %.15f %.15f %.15f\n',...
               channels_normal(x,58),channels_normal(x,59),...
               channels_normal(x,60),channels_normal(x,61),channels_normal(x,62),channels_normal(x,63));
        end
    end
    
end
fclose(train_x);
fclose(train_y);