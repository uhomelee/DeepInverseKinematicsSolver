temp=load('/home/student/DataPrepare/Comparison of different network structure/loss.txt');
x=temp(:,1)';
y=temp(:,2)';
plot(x,y,'g');
hold on;
temp=load('/home/student/DataPrepare/Comparison of different inputs/loss.txt');
x=temp(:,1)';
y=temp(:,2)';
plot(x,y,'r');
hold on;
temp=load('/home/student/DataPrepare/Comparison of different inputs/loss_5_10f.txt');
x=temp(:,1)';
y=temp(:,2)';
plot(x,y,'b');



