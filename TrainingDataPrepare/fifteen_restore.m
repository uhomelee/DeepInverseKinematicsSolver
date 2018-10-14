addpath('..//TrainingMaterial/matlab_lib');
addpath('..//TrainingMaterial/NDLUTIL0p161');
folders=dir('/home/nevershutdown/DataPrepare/Comparison of training and testing dataset/newchannelData/train');
for j=3:length(folders)
    channel_path=strcat('/home/nevershutdown/DataPrepare/Comparison of training and testing dataset/newchannelData/train/',folders(j).name)
    folder=strsplit(folders(j).name,'_');
    folder=folder(1)
    folder=folder{1};
    bvh_path=strcat('/home/nevershutdown/recent/data_full/',folder,'/',strrep(folders(j).name,'.txt',''),'.bvh')
    result_channel=load(channel_path);
    result_channel=result_channel.*180;
    %bvh_path=strcat('/home/student/fulldata/nine_input/oldbvh/',strrep(folders(j).name,'.txt',''),'.bvh')
    [skel,channels,frameLength] = bvhReadFile(bvh_path);
    for i=11:(size(channels,1)-10)
        channels(i,79)=result_channel(i-10,1);
        channels(i,80)=result_channel(i-10,2);
        channels(i,81)=result_channel(i-10,3);
        channels(i,82)=result_channel(i-10,4);
        channels(i,83)=result_channel(i-10,5);
        channels(i,84)=result_channel(i-10,6);
    end
    new_bvh=strcat('/home/nevershutdown/DataPrepare/Comparison of training and testing dataset/newbvh/train/',strrep(folders(j).name,'.txt',''),'.bvh')
    bvhWriteFile(new_bvh, skel, channels,frameLength);
end

folders=dir('/home/nevershutdown/DataPrepare/Comparison of training and testing dataset/newchannelData/test');
for j=3:length(folders)
    channel_path=strcat('/home/nevershutdown/DataPrepare/Comparison of training and testing dataset/newchannelData/test/',folders(j).name);
    folder=strsplit(folders(j).name,'_');
    folder=folder(1)
    folder=folder{1};
    bvh_path=strcat('/home/nevershutdown/recent/data_full/',folder,'/',strrep(folders(j).name,'.txt',''),'.bvh')
    result_channel=load(channel_path);
    result_channel=result_channel.*180;
    %bvh_path=strcat('/home/student/fulldata/nine_input/oldbvh/',strrep(folders(j).name,'.txt',''),'.bvh')
    [skel,channels,frameLength] = bvhReadFile(bvh_path);
    for i=11:(size(channels,1)-10)
        channels(i,79)=result_channel(i-10,1);
        channels(i,80)=result_channel(i-10,2);
        channels(i,81)=result_channel(i-10,3);
        channels(i,82)=result_channel(i-10,4);
        channels(i,83)=result_channel(i-10,5);
        channels(i,84)=result_channel(i-10,6);
    end
    new_bvh=strcat('/home/nevershutdown/DataPrepare/Comparison of training and testing dataset/newbvh/test/',strrep(folders(j).name,'.txt',''),'.bvh')
    bvhWriteFile(new_bvh, skel, channels,frameLength);
end