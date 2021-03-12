#### Autonomous Vehicles Team 5 Obstacle Avoidance

Ka Ming Chan
Evan Kim
Joseph Fallon

This package includes the code for calculating the SSIMs for the images taken by the ZED camera and Intel RealSense D455 camera and their ground truth images. Detailed explanations of the code are described in the Jupyter Notebook in the "notebooks" folder.

Instruction: 

If you would like to generate the SSIM results between an image and its ground truth image for each of the five filters, please simply run the command "python run.py ssim-text". The results will be written on the text file inside "results" folder.

If you would like to run the code and generate the plot results that are described in the report, please run the command "python run.py ssim-graph". The results will be saved as .png files inside "results" folder.

The command "python run.py test" will automatically generate all of the results listed above, inside the "results" folder.

Here is a Google Drive link to all of the images that our team has collected:
https://drive.google.com/drive/folders/1olPWEXQ5X8jmgcVDhKkoeA07meu2znwE?usp=sharing

The DockerHub Repository is:
https://hub.docker.com/repository/docker/kw24032/dsc180a-methodology7
