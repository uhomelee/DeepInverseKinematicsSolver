addpath('/home/student/recent/lib');
addpath('/home/student/recent/NDLUTIL0p161/NDLUTIL0p161');
folders=dir('/home/student/fulldata/src4/channel14');
for j=3:(length(folders)-1)
    channel_path=strcat('/home/student/fulldata/src4/channel14/',folders(j).name)
    result_channel=load(channel_path);
    result_channel=result_channel.*180;
    bvh_path=strcat('/home/student/fulldata/src4/oldbvh/',strrep(folders(j).name,'.txt',''),'.bvh')
    [skel,channels,frameLength] = bvhReadFile(bvh_path);
    for i=1:size(channels,1)
        channels(i,79)=result_channel(i,1);
        channels(i,80)=result_channel(i,2);
        channels(i,81)=result_channel(i,3);
        channels(i,82)=result_channel(i,4);
        channels(i,83)=result_channel(i,5);
        channels(i,84)=result_channel(i,6);
    end
    new_bvh=strcat('/home/student/fulldata/src4/newbvh14/',strrep(folders(j).name,'.txt',''),'.bvh');
    bvhWriteFile(new_bvh, skel, channels,frameLength);
end
