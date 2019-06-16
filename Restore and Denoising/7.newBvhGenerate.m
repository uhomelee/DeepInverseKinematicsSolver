addpath('../TrainingMaterial/matlab_lib/lib');
addpath('../TrainingMaterial/matlab_lib//NDLUTIL0p161');
folders=dir('./angleData');
for j=3:length(folders)
    channel_path_1=strcat('./angleData/',folders(j).name)
    channel_path_2=strcat('./angleData_2/',folders(j).name)
    folder=strsplit(folders(j).name,'_');
    folder=folder(1);
    folder=folder{1};
    bvh_path=strcat('../TrainingMaterial/bvh_file/data_full/',folder,'/',strrep(folders(j).name,'.txt',''),'.bvh')
    result_channel_1=load(channel_path_1);
    result_channel_2=load(channel_path_2);
    result_channel_1=result_channel_1.*180;
    result_channel_2=result_channel_2.*180;
    [skel,channels,frameLength] = bvhReadFile(bvh_path);
    for i=16:(size(channels,1)-10)
        channels(i,79)=(result_channel_1(i-15,1)+result_channel_1(i-14,1)+result_channel_1(i-13,1)+result_channel_1(i-12,1)+result_channel_1(i-11,1)+result_channel_1(i-10,1)...
            +result_channel_2(i-10,1)+result_channel_2(i-10,7)+result_channel_2(i-10,13)+result_channel_2(i-10,19)+result_channel_2(i-10,25))/11;
        channels(i,80)=(result_channel_1(i-15,2)+result_channel_1(i-14,2)+result_channel_1(i-13,2)+result_channel_1(i-12,2)+result_channel_1(i-11,2)+result_channel_1(i-10,2)...
            +result_channel_2(i-10,2)+result_channel_2(i-10,8)+result_channel_2(i-10,14)+result_channel_2(i-10,20)+result_channel_2(i-10,26))/11;
        channels(i,81)=(result_channel_1(i-15,3)+result_channel_1(i-14,3)+result_channel_1(i-13,3)+result_channel_1(i-12,3)+result_channel_1(i-11,3)+result_channel_1(i-10,3)...
            +result_channel_2(i-10,3)+result_channel_2(i-10,9)+result_channel_2(i-10,15)+result_channel_2(i-10,21)+result_channel_2(i-10,27))/11;
        channels(i,82)=(result_channel_1(i-15,4)+result_channel_1(i-14,4)+result_channel_1(i-13,4)+result_channel_1(i-12,4)+result_channel_1(i-11,4)+result_channel_1(i-10,4)...
            +result_channel_2(i-10,4)+result_channel_2(i-10,10)+result_channel_2(i-10,16)+result_channel_2(i-10,22)+result_channel_2(i-10,28))/11;
        channels(i,83)=(result_channel_1(i-15,5)+result_channel_1(i-14,5)+result_channel_1(i-13,5)+result_channel_1(i-12,5)+result_channel_1(i-11,5)+result_channel_1(i-10,5)...
            +result_channel_2(i-10,5)+result_channel_2(i-10,11)+result_channel_2(i-10,17)+result_channel_2(i-10,23)+result_channel_2(i-10,29))/11;
        channels(i,84)=(result_channel_1(i-15,6)+result_channel_1(i-14,6)+result_channel_1(i-13,6)+result_channel_1(i-12,6)+result_channel_1(i-11,6)+result_channel_1(i-10,6)...
            +result_channel_2(i-10,6)+result_channel_2(i-10,12)+result_channel_2(i-10,18)+result_channel_2(i-10,24)+result_channel_2(i-10,30))/11;
    end
    new_bvh=strcat('./newbvh/',strrep(folders(j).name,'.txt',''),'.bvh')
    bvhWriteFile(new_bvh, skel, channels,frameLength);
end