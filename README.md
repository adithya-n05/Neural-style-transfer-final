# Neural-Style-Transfer

This is a Keras implementation of Neural Style Transfer with Total Variation loss on a TensorFlow Backend. 

This implementation is based off of the papers "A Neural Algorithm of Artistic Style - Gatys et.al" and "Understanding Deep Image Representations by Inverting Them - Mahendran et.al".

# Configuring Style Transfer

It is recommended you run this on a GPU due to speed limits.

Change the variables content_img_path, style_img_path and save_directory to the paths of the content image and style image and the save directory along with initials for each output in the file "Neural_style_transfer_final.py" located in /Neural-Style-Transfer/Algorithms/Style transfer algorithms/Final implementation.

Change the c_weight, s_weight and tv_weight to your preference (default are 0.00002, 3200 and 400 respectively).

# Running Style Transfer

Run the script "Neural_style_transfer_final.py" within the command line by navigating to the directory where the file is stored and running:

```
python Neural_style_transfer_final.py
```
