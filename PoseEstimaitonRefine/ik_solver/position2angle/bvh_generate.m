addpath('../matlab_lib')
%retore the bvh before denoising
bvh_path='../original_fix.bvh'
[skel,channels,frameLength] = bvhReadFile(bvh_path);
new_channel=load('./angle_data.txt');
new_channel=new_channel*180;
for i =11:size(channels,1)-10
    %channels(i,126)=new_channel(i-10,1);
    %channels(i,125)=new_channel(i-10,2);
    %channels(i,124)=new_channel(i-10,3);
    channels(i,129)=new_channel(i-10,4);
    channels(i,128)=new_channel(i-10,5);
    channels(i,127)=new_channel(i-10,6);
end
new_bvh='./bvh_nodenoising.bvh'
bvhWriteFile(new_bvh, skel, channels,frameLength);

new_bvh='/home/nevershutdown/recent/data_full/01/01_01.bvh';
[skel,channels,frameLength] = bvhReadFile(new_bvh);

addpath('../matlab_lib')
%retore the bvh before denoising
bvh_path='../original.bvh'
[skel,channels,frameLength] = bvhReadFile(bvh_path);
new_channel=load('./angle_data.txt');
new_channel=new_channel*180;
for i =11:size(channels,1)-10
    %channels(i,126)=new_channel(i-10,1);
    %channels(i,125)=new_channel(i-10,2);
    %channels(i,124)=new_channel(i-10,3);
    channels(i,129)=new_channel(i-10,4);
    channels(i,128)=new_channel(i-10,5);
    channels(i,127)=new_channel(i-10,6);
end
new_bvh='./bvh_nodenoising_right_hand.bvh'
bvhWriteFile(new_bvh, skel, channels,frameLength);
