# Pixel Level SSIM Computation

Here I have tried to perform pixel level computation only on tumor region.

* I obtain tumor region with help of mask.
* Flatten the binary array.
* Filter only non-zero pixels.
* Compute SSIM for this flattened out array.

## Reference
I have used following code reference to come up with keras based pixel level ssim calcualtion.
* [SSIM-PyTorch](https://github.com/pranjaldatta/SSIM-PyTorch/blob/master/SSIM_notebook.ipynb)
* [CV Notes](https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python)
