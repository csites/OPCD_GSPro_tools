<P>You need to downloads SAM's Checkpoint files here.    I recommend the VIT_B to train on, its faster to train and better for the small memory GPUs.   It's also the smallest of the pre-trained sets at 375MB.  If you want to run SAM standalone (sam1_svg10.py) without any training or finetuning, then I recommend you pull down the huge model (VIT_H). It works good for a starting point for segmenting the golf courses.    vit_b = SMALL, vit_l = LARGE, vit_h = HUGE!
</P>
<P> This is up-to-date as of 04/28/2025.  I really recommend you goto https://github.com/facebookresearch/segment-anything and read their page.
</P>

<A HREF="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth">DOWNLOAD VIT_H="</A>
<A HREF="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth">DOWNLOAD VIT_L="</A>
<A HREF="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth">DOWNLOAD VIT_B="</A>

Contraints with Github filesize limits  prevent me from including these file or the example finetuning checkpoints I've made.  

