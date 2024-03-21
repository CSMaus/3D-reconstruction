# 3D-reconstruction
Attempts to create 3D reconstruction app (web or not) for small things (not rooms)

# How to install CUDA

CUDA installation guide:

FIRST OF ALL: Check if your GPU is support CUDA, then follow this steps:
1. Nvidia driver for your GPU: https://www.nvidia.com/download/index.aspx
2. CUDA 11.6 (for example): https://marketplace.visualstudio.com/items?itemName=NVIDIA.NvNsightToolsVSIntegration
       And install it.
3. cuDNN archive 8.6 for 11.6 (compatible, for example). After installing CUDA 11.6, copy and paste from archive **bin/lib/include** folders all files into `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6` in corresponding **bin/lib/include** folders.
4. NVIDIA Nsight Integration (64-bit): https://marketplace.visualstudio.com/items?itemName=NVIDIA.NvNsightToolsVSIntegration
5. run in cmd (if you already installed python): `pip3 install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116`

<br><br> *NOTE*: if Visual Studio is not installed, you mught need to install it with C++ packages.<br>

# Back to 3D-reconstruction
### Creating depth map in reil-time using CPU or cuda 
(depends on what you have)
Using simple algorithm similar procedural height map generation and then terrain 
(landscape) generation, create 3D point cloud from depth map.
In future, I will add ability to create mesh from depth map which is better for many application.
<br>
Here is original video input and depth map for it calculated using midas library "MiDaS_small" Neural Network.
Later modifications will be added to create accurate 3D for object based on video of it (not just one frame as it does now)
<br>
![3D_rec/Gifs/output.gif](3D_rec/Gifs/output.gif)
![img_1.png](img_1.png)
![img.png](img.png)
