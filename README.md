<h1># OPCD_GSPro_tools</h1></B>
<h2>Tools for building GSPro golf courses</h2>
<p>
<H3>INTRO:</H3>
</p>
<p>These are some tools I've been working on to help in the mapping of golf courses to build course models for GSPro.   These are all pretty heavy programs segmentation programs or segmentation training programs written in python3 on ubuntu and use CUDA, Pytorch, and Meta's 'Segment_Anything'.  It uses tensorflow extensively and is very heavy on the GPU.   While you can run the segmentation programs on cpu they can take hours even on the beafiest CPUs.  I estimated that on my Ryzen 9 5950X at 4.5Ghz, it was still going to take 10 days to run the training program where as with a simple Nvidia RTX 4060 it would be about 6hrs.   Initially I tried running on ROCM on and AMD 6900RX but that got complicated and broken really fast.  So this code may never work for the AMD GPU cards sadly. </p>  
<p> The first program is used to read the OPCD/QGIS/Overlays images. These are supposed to be 8192x8192 jpg or tiff images you downloaded from Google Earth or Bing.  Due to memory limitations I resize these images down to 1892x1892 to fit the RTX 4060's memory limitation (8GB). It can fit 2048x2048 but with Ubuntu and KDE running sometimes some of the GPU memory is consumed by them.  You might want to experiment with the scaling if you have a better GPU and I recommend it.  The first program is called sam1_svg10.py (my 10th interation of this program).   I had a lot of AI help with this from Google Gemini which just blows my mind how good it has become at python coding.  I mean really good!  I never new all the details about SVG images or how to process them.   The initial inspiration for this was from a demonstration of Meta's Segment_Anything model, which was one of the coolest demos I've seen.   As background I worked for years at the Computer Vision and Image Processing Laboratory and did a lot of work in segmentation and It seemed to me that segmenting a golf course could help the processes of hand classification of arial images in Inkscape.  It doesn't get rid if the Inkscape step, but it does help.</p>
<p>The other program is also a very complex bit of python code to train SAM to classify golf courses.  On my Nvidia RTX 4060 this took about 6.5 hours to run and it generated a special vit_b model sam_finetuned_golf_epoch_XX.pth that should find the most significant parts of a golf course.  The model is based on the dataset from the work of VINÍCIUS SOARES MATTHIESEN, on "Semantic Segmentation of Danish Golf Courses U-Net" https://www.kaggle.com/code/viniciussmatthiesen/semantic-segmentation-of-danish-golf-courses-u-net.  I obtained a lot of ideas from that code.  Kaggle is a good learn tool I have to say.  If your a GSPro Course developer and would like to contribute your manually developed Inkscape course classification images along with the original Overlay images used, this model might be able to be improved. It's like AI, the more it's trained the better the segmentation will be.</p>
<p>Anyway, that is what this project is about; to develope a trained model for SAM that can be used in a python program to segment and classify the features of a golf course from an arial (satellite image) as found in the Google or Bing maps. </p>
<p>Two programs are presented here;  train_sam2.py is the program that uses the course data from U-Net training set to build a the SAM checkpoint file: sam_finetuned_golf_epoch_XX.pth. The second program is sam1_svg10.py which uses the checkpoint file and SAM to segment your golf course images into classes and creates a SVG image of the original course as the background image with an overlay of filled with a opaque shape of the feature that was segmented out of the image.   You simply need to delete the mislabled (mis-identified) shape items, and refill the items you see with the proper colors for the OPCD tools used later in the Blender step (the step the imports the height maps).  </p>
<p></p>Steps to build and run these:   First your hardware.  Somebody may be able to get this working on AMD's GPUs with ROCm but it wasn't mature enough for me. (if you do fork this and get it running on an AMD GPU I would love to see it).   So you need a Nvidia RTX 4060 or better if you want to keep your sanity.  I use Kubuntu 24.10, python3 and venv (python conda would also work well for your version managment).  I personally like venv with linux because it's everything is kept in the user's ~/venv folder (libraries and everything) and not the system's folders.   It makes it easy to move to other users and contain everything.   You will then need to run ~/venv/bin/pip install -r requirements.txt.   Obviously you will want to review the requirements file.   This requirements.txt has many pieces.  You will also need to make sure that your Ubuntu (Kubuntu) has support for your Nvidia adapter.  For me I needed to do this; </p>

<pre><code>
# Update the ubuntu repo keys (as root or prefix every line with 'sudo ')
wget -O /etc/apt/keyrings/cuda-keyring_1.1-1_all.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2410/x86_64/cuda-keyring_1.1-1_all.deb

apt install nvidia-cuda-dev nvidia-cuda-toolkit nvidia-cuda-samples nvitop nvidia-cuda-toolkit-doc 
apt install nvidia-gds
apt install nvidia-cuda-toolkit
apt install nvidia-cuda-dev
apt install hipcc-nvidia-rpath6.3.1 hipcc-nvidia6.3.1 hipcc-nvidia-dbgsym hipcc-nvidia 
apt install cudnn9-cuda-12-8 libcudnn-frontend-dev libcudnn9-static-cuda-12 libcutlass-dev nvidia-cudnn

# Optional Docker GPU stuff for like ollama and things. 
apt-get install -y nvidia-container-toolkit

# Note; version numbers may change with time. 

If that all looks good then we need to get the SAM libraries and the vit_h and vit_b checkpoint files specifically.  Please refer to these sites on SAM;

* SAM 2 code: https://github.com/facebookresearch/segment-anything-2
* SAM 2 demo: https://sam2.metademolab.com/
* SAM 2 paper: https://arxiv.org/abs/2408.00714
</code></pre>

<p> Do visit the site https://github.com/facebookresearch/segment-anything.   It is an amazing cool tool and you will probably understand using SAM to segment golf course images.  Also do donwload that code base.  It's fun to play with in 'jupygter notebook' on Ubuntu.  You will need to download a 'vit_ SAM model listed in the SAM documentation</p>
<pre>
    <code>
        SMALL lightweight: vit_b -> https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
        Medium:  vit_l -> https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
        Heavy: vit_h -> https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    </code>
</pre>

# Usage

*(This section would detail how to prepare input images and run each scripts)*

---

## Credits

* Special thanks to Google Gemini for invaluable assistance with Python coding and understanding complex libraries like `segment-anything` and SVG processing.
* Initial inspiration and dataset context from the work of Vinícius Soares Matthiesen on "Semantic Segmentation of Danish Golf Courses U-Net" ([https://www.kaggle.com/code/viniciussmatthiesen/semantic-segmentation-of-danish-golf-courses-u-net](https://www.kaggle.com/code/viniciussmatthiesen/semantic-segmentation-of-danish-golf-courses-u-net)).
* Based on the original Segment Anything Model (SAM) by Meta AI ([https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)).

---

## License

GPLv3.  Basically if you have a change to the basic code that works for you, I would love to see or hear about it.  
