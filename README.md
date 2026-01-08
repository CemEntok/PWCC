# PWCC: Pixel-wise Color Constancy via Smoothing Techniques in Multi-Illuminant Scenes
---

This repository includes the implementation details of Pixel-wise Color Constancy (PWCC) proposed in **arXiV**:

**Pixel-wise Color Constancy via Smoothing Techniques in Multi-Illuminant Scenes**

![SCC_Pipeline drawio](https://github.com/CemEntok/PWCC/assets/97525722/229b1426-ca18-45af-b333-4bb16f748760)

*PWCC learns a pixel-wise mapping, referred to as an illuminant map, between input and ground truth images. This mapping provides a white-balancing operation to the input image so that color casts can be minimized.*

---

Conda environment setup:
```
# In Linux Environment:
$ while read requirement; do conda install --yes $requirement || pip install $requirement; done < envcc.txt
```

# Citation

If you use the provided method in this repository, please cite the following article:

```

@INPROCEEDINGS{10647547,
  author={Entok, Umut Cem and Laakom, Firas and Pakdaman, Farhad and Gabbouj, Moncef},
  booktitle={2024 IEEE International Conference on Image Processing (ICIP)}, 
  title={Pixel-Wise Color Constancy Via Smoothness Techniques In Multi-Illuminant Scenes}, 
  year={2024},
  volume={},
  number={},
  pages={2737-2743},
  keywords={Training;Reflectivity;Uncertainty;Smoothing methods;Image color analysis;Lighting;Estimation;Color Constancy;Multi-Illuminant Estimation;Total Variation;Bilateral Filter;Label-smoothing},
  doi={10.1109/ICIP51287.2024.10647547}}

}
```
## Downloading the Dataset

This paper uses LISMI Dataset. Please follow the instructions of the study employed this dataset from here: [LSMI Dataset Download](https://github.com/DY112/LSMI-dataset?tab=readme-ov-file#download-the-lsmi-dataset)



