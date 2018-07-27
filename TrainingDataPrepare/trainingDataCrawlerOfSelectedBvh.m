%prepare new 15 input train data after select bad data from test sample
format long;
addpath('/home/student/recent/lib');
addpath('/home/student/recent/NDLUTIL0p161/NDLUTIL0p161');
train_x=fopen('/home/student/train_x_5_10f_selected.txt','a');
train_y=fopen('/home/student/train_y_5_10f_selected.txt','a');
folder='/home/student/recent/train';
files=dir(folder);
for j=3:length(files)
    file_add=strcat(folder,'/',files(j).name)
    [skel,channels,frameLength] = bvhReadFile(file_add);
        %calculate the arm
    xyz_temp=bvh2xyz(skel,channels(1,:));
    length_arm=norm(xyz_temp(33,:)-xyz_temp(32,:))+norm(xyz_temp(32,:)-xyz_temp(31,:));
    channels_normal=channels/180;
    for x=11:(size(channels,1)-10)
            %normalize xyz
           xyz_full= bvh2xyz(skel, channels(x,:));
           xyz=(xyz_full(33,:)-xyz_full(31,:))./length_arm;
           
           xyz_full_pre= bvh2xyz(skel, channels(x-10,:));
           xyz_pre=(xyz_full_pre(33,:)-xyz_full_pre(31,:))./length_arm;
           
           xyz_full_beh= bvh2xyz(skel, channels(x+10,:));
           xyz_beh=(xyz_full_beh(33,:)-xyz_full_beh(31,:))./length_arm;
           
           xyz_full_pre2= bvh2xyz(skel, channels(x-5,:));
           xyz_pre2=(xyz_full_pre2(33,:)-xyz_full_pre2(31,:))./length_arm;
           
           xyz_full_beh2= bvh2xyz(skel, channels(x+5,:));
           xyz_beh2=(xyz_full_beh2(33,:)-xyz_full_beh2(31,:))./length_arm;
           
           fprintf(train_x,'%.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f\n',...
               xyz_pre(1),xyz_pre(2),xyz_pre(3),...
               xyz_pre2(1),xyz_pre2(2),xyz_pre2(3),...
               xyz(1),xyz(2),xyz(3),...
               xyz_beh2(1),xyz_beh2(2),xyz_beh2(3),...
               xyz_beh(1),xyz_beh(2),xyz_beh(3));
           fprintf(train_y,'%.15f %.15f %.15f %.15f %.15f %.15f\n',...
               channels_normal(x,79),channels_normal(x,80),...
               channels_normal(x,81),channels_normal(x,82),channels_normal(x,83),channels_normal(x,84));
     end
end
fclose(train_x);
fclose(train_y);