# PWCC: Pixel-wise Color Constancy via Smoothing Techniques in Multi-Illuminant Scenes
---

This repository includes the implementation details of Pixel-wise Color Constancy (PWCC) proposed in **arXiV**:

**Pixel-wise Color Constancy via Smoothing Techniques in Multi-Illuminant Scenes**

![SCC_Pipeline drawio](https://github.com/CemEntok/PWCC/assets/97525722/229b1426-ca18-45af-b333-4bb16f748760)

*PWCC learns a pixel-wise mapping, referred to as an illuminant map, between input and ground truth images. This mapping provides a white-balancing operation to the input image so that color casts can be minimized.*

Conda environment setup:
```
# In Linux Environment:
$ while read requirement; do conda install --yes $requirement || pip install $requirement; done < envcc.txt
```
