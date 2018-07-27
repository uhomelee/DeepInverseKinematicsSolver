addpath('/home/student/recent/lib');
addpath('/home/student/recent/NDLUTIL0p161/NDLUTIL0p161');
folders=dir('/home/student/fulldata/nine_input/channel_5_10f');
for j=3:length(folders)
    channel_path=strcat('/home/student/fulldata/nine_input/channel_5_10f/',folders(j).name)
    result_channel=load(channel_path);
    result_channel=result_channel.*180;
    bvh_path=strcat('/home/student/fulldata/nine_input/oldbvh/',strrep(folders(j).name,'.txt',''),'.bvh')
    [skel,channels,frameLength] = bvhReadFile(bvh_path);
    for i=11:(size(channels,1)-10)
        channels(i,79)=result_channel(i-10,1);
        channels(i,80)=result_channel(i-10,2);
        channels(i,81)=result_channel(i-10,3);
        channels(i,82)=result_channel(i-10,4);
        channels(i,83)=result_channel(i-10,5);
        channels(i,84)=result_channel(i-10,6);
    end
    new_bvh=strcat('/home/student/fulldata/nine_input/newbvh_5_10f/',strrep(folders(j).name,'.txt',''),'.bvh');
    bvhWriteFile(new_bvh, skel, channels,frameLength);
end