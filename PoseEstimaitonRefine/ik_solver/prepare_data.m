addpath('./matlab_lib')
%enlarge offset
bvh_path=strcat('./estimated_animation.bvh');
[skel,channels,frameLength] = bvhReadFile(bvh_path);
for i =1:81
    skel.tree(i).offset=skel.tree(i).offset*10;
end
new_bvh='./original.bvh'
bvhWriteFile(new_bvh, skel, channels,frameLength);
%%
%smooth the original bvh
bvh_path='./original.bvh'
[skel,channels,frameLength] = bvhReadFile(bvh_path);
for i = 11:(size(channels,1)-10)
    for j =1:size(channels,2)
        channels(i,j)=(channels(i-10,j)+channels(i-8,j)+channels(i-6,j)+channels(i-4,j)+channels(i-2,j)+channels(i,j)+...
            channels(i+2,j)+channels(i+4,j)+channels(i+6,j)+channels(i+8,j)+channels(i+10,j))/11;
    end
end
new_bvh2='./original_fix.bvh'
bvhWriteFile(new_bvh2,skel,channels,frameLength);

%smooth the original bvh
bvh_path='./original.bvh'
[skel,channels,frameLength] = bvhReadFile(bvh_path);
for i = 11:(size(channels,1)-10)
    for j =1:size(channels,2)
        channels(i,j)=(channels(i-5,j)+channels(i-4,j)+channels(i-3,j)+channels(i-2,j)+channels(i-1,j)+channels(i,j)+...
            channels(i+1,j)+channels(i+2,j)+channels(i+3,j)+channels(i+4,j)+channels(i+5,j))/11;
    end
end
new_bvh2='./original_fix.bvh'
bvhWriteFile(new_bvh2,skel,channels,frameLength);
%%
%prepare position data for predict angle
%%%%
addpath('./matlab_lib')
bvh_path='./original_fix.bvh'
[skel,channels,frameLength] = bvhReadFile(bvh_path);
xyz_temp=bvh2xyz(skel,channels(1,:));
length_arm=norm(xyz_temp(39,:)-xyz_temp(38,:))+norm(xyz_temp(38,:)-xyz_temp(37,:));
file_path='./position2angle/position_data.txt';
fwrite=fopen(file_path,'a');
for x=11:(size(channels,1)-10)
    xyz_full= bvh2xyz(skel, channels(x,:));
    xyz=(xyz_full(39,:)-xyz_full(37,:))./length_arm;
    
    xyz_full_pre= bvh2xyz(skel, channels(x-10,:));
    xyz_pre=(xyz_full_pre(39,:)-xyz_full_pre(37,:))./length_arm;
    
    xyz_full_pre2= bvh2xyz(skel, channels(x-5,:));
    xyz_pre2=(xyz_full_pre2(39,:)-xyz_full_pre2(37,:))./length_arm;
    
    xyz_full_pre3= bvh2xyz(skel, channels(x-1,:));
    xyz_pre3=(xyz_full_pre3(39,:)-xyz_full_pre3(37,:))./length_arm;
    
    xyz_full_beh= bvh2xyz(skel, channels(x+10,:));
    xyz_beh=(xyz_full_beh(39,:)-xyz_full_beh(37,:))./length_arm;
    
    xyz_full_beh2= bvh2xyz(skel, channels(x+5,:));
    xyz_beh2=(xyz_full_beh2(39,:)-xyz_full_beh2(37,:))./length_arm;
    
    xyz_full_beh3= bvh2xyz(skel, channels(x+1,:));
    xyz_beh3=(xyz_full_beh3(39,:)-xyz_full_beh3(37,:))./length_arm;
    fprintf(fwrite,'%.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f\n',...
        xyz_pre(3),xyz_pre(2),xyz_pre(1),...
        xyz_pre2(3),xyz_pre2(2),xyz_pre2(1),...
        xyz_pre3(3),xyz_pre3(2),xyz_pre3(1),...
        xyz(3),xyz(2),xyz(1),...
        xyz_beh3(3),xyz_beh3(2),xyz_beh3(1),...
        xyz_beh2(3),xyz_beh2(2),xyz_beh2(1),...
        xyz_beh(3),xyz_beh(2),xyz_beh(1));
end
fclose(fwrite);

%%
%%prepare the data of denoising
addpath('./matlab_lib')
bvh_path='./position2angle/bvh_nodenoising.bvh'
[skel,channels,frameLength] = bvhReadFile(bvh_path);
xyz_temp=bvh2xyz(skel,channels(1,:));
length_arm=norm(xyz_temp(39,:)-xyz_temp(38,:))+norm(xyz_temp(38,:)-xyz_temp(37,:));
file_path='./denoising/position_data.txt';
fwrite=fopen(file_path,'a');
for x =11:size(channels,1)-10
    xyz_full_pre= bvh2xyz(skel, channels(x-10,:));
    xyz_pre=(xyz_full_pre(39,:)-xyz_full_pre(37,:))./length_arm;
    
    xyz_full_pre2= bvh2xyz(skel, channels(x-8,:));
    xyz_pre2=(xyz_full_pre2(39,:)-xyz_full_pre2(37,:))./length_arm;
    
    xyz_full_pre3= bvh2xyz(skel, channels(x-6,:));
    xyz_pre3=(xyz_full_pre3(39,:)-xyz_full_pre3(37,:))./length_arm;
    
    xyz_full_pre4= bvh2xyz(skel, channels(x-4,:));
    xyz_pre4=(xyz_full_pre4(39,:)-xyz_full_pre4(37,:))./length_arm;
    
    xyz_full_pre5= bvh2xyz(skel, channels(x-2,:));
    xyz_pre5=(xyz_full_pre5(39,:)-xyz_full_pre5(37,:))./length_arm;
    
    xyz_full_pre6= bvh2xyz(skel, channels(x-1,:));
    xyz_pre6=(xyz_full_pre6(39,:)-xyz_full_pre6(37,:))./length_arm;
    
    xyz_full_pre7= bvh2xyz(skel, channels(x,:));
    xyz_pre7=(xyz_full_pre7(39,:)-xyz_full_pre7(37,:))./length_arm;
    fprintf(fwrite,'%.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f\n',...
        xyz_pre(3),xyz_pre(2),xyz_pre(1),...
        xyz_pre2(3),xyz_pre2(2),xyz_pre2(1),...
        xyz_pre3(3),xyz_pre3(2),xyz_pre3(1),...
        xyz_pre4(3),xyz_pre4(2),xyz_pre4(1),...
        xyz_pre5(3),xyz_pre5(2),xyz_pre5(1),...
        xyz_pre6(3),xyz_pre6(2),xyz_pre6(1),...
        xyz_pre7(3),xyz_pre7(2),xyz_pre7(1));
end
fclose(fwrite);

%%
%%denoising bvh generate
addpath('../matlab_lib')
bvh_path='../position2angle/bvh_nodenoising.bvh'
[skel,channels,frameLength] = bvhReadFile(bvh_path);
beh_channels=load('./angle_data.txt');
for i =11:size(channels,1)-10
   channels(i,124)=(channels(i-5,124)+channels(i-4,124)+channels(i-3,124)+channels(i-2,124)+channels(i-1,124)+channels(i,124)...
       +beh_channels(i-10,3)+beh_channels(i-10,9)+beh_channels(i-10,15)+beh_channels(i-10,21)+beh_channels(i-10,27))/11;
   channels(i,125)=(channels(i-5,125)+channels(i-4,125)+channels(i-3,125)+channels(i-2,125)+channels(i-1,125)+channels(i,125)...
       +beh_channels(i-10,2)+beh_channels(i-10,8)+beh_channels(i-10,14)+beh_channels(i-10,20)+beh_channels(i-10,26))/11;
   channels(i,126)=(channels(i-5,126)+channels(i-4,126)+channels(i-3,126)+channels(i-2,126)+channels(i-1,126)+channels(i,126)...
       +beh_channels(i-10,1)+beh_channels(i-10,7)+beh_channels(i-10,13)+beh_channels(i-10,19)+beh_channels(i-10,25))/11;
   channels(i,127)=(channels(i-5,127)+channels(i-4,127)+channels(i-3,127)+channels(i-2,127)+channels(i-1,127)+channels(i,127)...
       +beh_channels(i-10,6)+beh_channels(i-10,12)+beh_channels(i-10,18)+beh_channels(i-10,24)+beh_channels(i-10,30))/11;
   channels(i,128)=(channels(i-5,128)+channels(i-4,128)+channels(i-3,128)+channels(i-2,128)+channels(i-1,128)+channels(i,128)...
       +beh_channels(i-10,5)+beh_channels(i-10,11)+beh_channels(i-10,17)+beh_channels(i-10,23)+beh_channels(i-10,29))/11;
   channels(i,129)=(channels(i-5,129)+channels(i-4,129)+channels(i-3,129)+channels(i-2,129)+channels(i-1,129)+channels(i,129)...
       +beh_channels(i-10,4)+beh_channels(i-10,10)+beh_channels(i-10,16)+beh_channels(i-10,22)+beh_channels(i-10,28))/11;
end
new_bvh='./bvh_denoising.bvh'
bvhWriteFile(new_bvh,skel,channels,frameLength);













