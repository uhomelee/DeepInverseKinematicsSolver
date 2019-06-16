# Deep Inverse Kinematics Solver
>The problem of Inverse Kinematics is solved with deep neural network, trained with so-far the largest human mocap database.Given the challenge of multi-solution in the problem of IK, the trained model selects the pose which is most consistent with the pose by real performer.Such consistency is validated by the comparison with the benchmark database.
# Preparation
## Hardware Requirement
Those neural networks are trained and tested on a standard computer with OS:Ubuntu, CPU:Intel® Core™ i7-6800K CPU @ 3.40GHz × 12, GPU:Nvidia TITAN Xp, Memory: 16G. Since it cost about ***60 hours*** to train the networks basing on this equipment, ***the higher configuration and larger memory*** is better. 
## Software Requirement and Including Libraries
- Python 3
- Tensorflow(GPU version is better)
- Matlab
- Numpy etc.
# IK Solver Implemention
## Training & Testing data Preparation
*please download  bvh files through this [link](链接：https://pan.baidu.com/s/1c0sVuxSuR_6BHa11Oa8u0g) and extract password is ew8u. Then put the whole `data_full` folder into `TrainingCodePrepare->TrainingMaterial->bvh_file`*
prepare the training and testing dataset through run TrainingDataPrepare.m files in MATLAB or follow the commands below
```
cd TrainingDataPrepare
matlab -nodesktop -nosplash -r TrainingDataCrawl
cd ../TrainingCode
```
## Training Process
run the python files in `TrainingCode` folder through command line using the following statements to get a coarse prediction model.
```
python rnn.py
cd ../
```
or
```
pytohn rnn_multilayer.py
cd ../
```
code of comparions in experiments is put in folders in `TrainingCode` respectively. You can run the `.py` files to see the result.

## Denoising
run the MATLAB files and Python files to refine the result of original output. follow the commands below in command line one by one.
```
mkdir positionData
mkdir predictData
mkdir angleData
mkdir angleData_2
mkdir newbvh
cd Restore and Denoising
matlab -nodesktop -nosplash -r trainingDataCrawlForDenoising
python fixangle_rnn.py
matlab -nodesktop -nosplash -r positionDataPrepare
python restore_rnn.py
matlab -nodesktop -nosplash -r predictDataPrepare
python restore_rnn_denoising.py
matlab -nodesktop -nosplash -r newBvhGenerate
cd ../
```
The new bvh files aftering denoising are generated in `newbvh` folder, upload them to [bvhplayer](http://lo-th.github.io/olympe/BVH_player.html)to see the result of IK solver.

## Refine Pose Estimation Result
### install liabraries, initial imports and initializations
Firstly, install liabraries, initial imports and initializations in command line `window1`
```
pip install configobj
pip install ffmpeg
apt-get install blender
```
then download the pose estimation model
```
cd PoseEstimationRefine
wget https://people.eecs.berkeley.edu/~kanazawa/cachedir/hmr/models.tar.gz && tar -xf models.tar.gz
mv models hmr/
pip install -r hmr/requirements.txt
mkdir hmr/output
mkdir hmr/output/csv
mkdir hmr/output/images
mkdir hmr/output/csv_joined
mkdir hmr/output/bvh_animation
cd keras_Realtime_Multi_Person_Pose_Estimation
wget -nc --directory-prefix=./keras/ 		https://www.dropbox.com/s/llpxd14is7gyj0z/model.h5
mkdir sample_jsons
mkdir sample_videos
mkdir sample_images
```
then open a new command line windows `window2` to run python commands
```
python model_load.py
```
### upload video
secondly, put the video file in `sample_videos`
### process the video
 thirdly, process video in command line `window1`
```
cd ../
bash hmr/3dpose_estimate.sh
blender --background hmr/csv_to_bvh.blend -noaudio -P hmr/csv_to_bvh.py
```
the bvh files are generated in folder `hmr/output/bvh_animation/`
### refine the result through ik solver
you need to move the model trained in above steps to model folder respectively.
```
mkdir ik_solver/denoising/model
mkdir ik_solver/position2angle/model
```
fourthly, copy the bvh file generated to `ik_solver` then run the `prepare_data.m`. After that, run the `.py` files in `positon2angle` and `denoising` folders respectively. The final result is in `denosing` folder. 