%preapare xyz_data of foot(5frame)
addpath('/home/student/recent/lib');
addpath('/home/student/recent/NDLUTIL0p161/NDLUTIL0p161');
format long;
folders=dir('/home/student/recent/data_full');
for i=3:length(folders)
%for i=3:12
    folder=strcat('/home/student/recent/data_full/',folders(i).name);
    files=dir(folder);
    for j=3:floor(length(files)/10*9)
        file_add=strcat(folder,'/',files(j).name)
        [skel,channels,frameLength] = bvhReadFile(file_add);
        file_name=strrep(files(j).name,'.bvh','');
        file_path=strcat('/home/student/DataPrepare/Comparison of foot and arm/xyzData/',file_name,'.txt')
        fwrite=fopen(file_path,'a');
        xyz_temp=bvh2xyz(skel,channels(1,:));
        length_foot=norm(xyz_temp(11,:)-xyz_temp(10,:))+norm(xyz_temp(10,:)-xyz_temp(9,:));
        channels_normal=channels/180;
        for x=11:(size(channels,1)-10)
            %normalize xyz
           xyz_full= bvh2xyz(skel, channels(x,:));
           xyz=(xyz_full(11,:)-xyz_full(9,:))./length_foot;
           
           xyz_full_pre= bvh2xyz(skel, channels(x-10,:));
           xyz_pre=(xyz_full_pre(11,:)-xyz_full_pre(9,:))./length_foot;
           
           xyz_full_beh= bvh2xyz(skel, channels(x+10,:));
           xyz_beh=(xyz_full_beh(11,:)-xyz_full_beh(9,:))./length_foot;
           
           xyz_full_pre2= bvh2xyz(skel, channels(x-5,:));
           xyz_pre2=(xyz_full_pre2(11,:)-xyz_full_pre2(9,:))./length_foot;
           
           xyz_full_beh2= bvh2xyz(skel, channels(x+5,:));
           xyz_beh2=(xyz_full_beh2(11,:)-xyz_full_beh2(9,:))./length_foot;
           
           fprintf(fwrite,'%.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f\n',...
               xyz_pre(1),xyz_pre(2),xyz_pre(3),...
               xyz_pre2(1),xyz_pre2(2),xyz_pre2(3),...
               xyz(1),xyz(2),xyz(3),...
               xyz_beh2(1),xyz_beh2(2),xyz_beh2(3),...
               xyz_beh(1),xyz_beh(2),xyz_beh(3));
        end
        fclose(fwrite);
    end
   %{
    for j=(floor(length(files)/10*9)+1):length(files)
        file_add=strcat(folder,'/',files(j).name)
        [skel,channels,frameLength] = bvhReadFile(file_add);
        file_name=strrep(files(j).name,'.bvh','');
        file_path=strcat('/home/student/15input_xyz_5&10/test/',file_name,'.txt')
        fwrite=fopen(file_path,'a');
        xyz_temp=bvh2xyz(skel,channels(1,:));
        length_arm=norm(xyz_temp(33,:)-xyz_temp(32,:))+norm(xyz_temp(32,:)-xyz_temp(31,:));
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
           
           fprintf(fwrite,'%.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f\n',...
               xyz_pre(1),xyz_pre(2),xyz_pre(3),...
               xyz_pre2(1),xyz_pre2(2),xyz_pre2(3),...
               xyz(1),xyz(2),xyz(3),...
               xyz_beh2(1),xyz_beh2(2),xyz_beh2(3),...
               xyz_beh(1),xyz_beh(2),xyz_beh(3));
        end
        fclose(fwrite);
    end
    %}
end