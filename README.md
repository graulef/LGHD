This is a C++ implementation of the Log-Gabor Histogram descriptor (LGHD) proposed by Cristhian Aguilera and Angel D. Sappa and Ricardo Toledo in 2015. For further information on their work, please refer to the LGHD section below or to https://github.com/ngunsu/LGHD.

The code itself is largely based on a C++ implementation of Log-Gabor-based Phase Congruency generation by Carlos H Villa Pinto. For further information on his implementation and install instructions for dependencies, please refer to the Phase Congurency section below or to https://github.com/chvillap/phase-congruency-features. 

### LGHD

**Article abstract**:
This paper presents a new feature descriptor suitable to the task of matching features points between images with non-linear intensity variations. This includes image pairs with significant illuminations changes, multi-modal image pairs and multi-spectral image pairs. The proposed method describes the neighbourhood of feature points combining frequency and spatial information using multi-scale and multi-oriented Log-Gabor filters.
Experimental results show the validity of the proposed approach and also the improvements with respect to the state of the art.

**Article url**: [link](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=7350783&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D7350783)

**Bibtex**
```
@inproceedings{lghd2015,
  organization = { IEEE },
  year = { 2015 },
  pages = { 5 },
  month = { Sep },
  location = { Quebec, Canada },
  booktitle = { Image Processing (ICIP), 2015 IEEE International Conference on },
  author = { Cristhian Aguilera and Angel D. Sappa and Ricardo Toledo },
  title = { LGHD: A feature descriptor for matching across non-linear intensity variations },
}
```
**Datasets used in the article**

* LWIR/RGB (Ours) https://owncloud.cvc.uab.es/owncloud/index.php/s/1Wx715yUh6kDAO7

* LWIR/NIR  http://ivrl.epfl.ch/supplementary_material/cvpr11/

* FLASH/NO-FLASH and RGB/DEPTH http://www.cse.cuhk.edu.hk/leojia/projects/multimodal/index.html


### Phase Congruency

C++ implementation of the phase congruency technique for 2D and 3D data.

**Phase congruency** is a signal processing technique better known for its use in the detection of invariant image features. Rather than assuming that an image should be compressed into a set of edges, the phase congruency model assumes that the compressed image format should be high in information (or entropy), and low in redundancy. Thus, instead of searching for points where there are sharp changes in intensity, this model searches for patterns of order in the phase component of the Fourier transform. Phase is chosen because experiments demonstrated that it is crucial to the perception of visual features. Further physiological evidence indicates that the human visual system responds strongly to points in an image where the phase information is highly ordered. Thus the phase congruency model defines features as points in an image with high phase order. And for that it has a series of advantages over other image feature detectors.

The phase congruency measure is proportional to the local energy of the signal, therefore it can be calculated via convolution of the original image with a bank of spatial filters in quadrature. A bank of [log-Gabor filters](https://en.wikipedia.org/wiki/Log_Gabor_filter) is especially suited for that, thus an implementation of 2D/3D log-Gabor filters is also part of this project.

More details about phase congruency and some of its applications can be found in the following papers:

> Kovesi, P., 2000. Phase congruency: A low-level image invariant. Psychological Research 64, 136-148.

> Kovesi, P., 2003. Phase congruency detects corners and edges, in: The Australian Pattern Recognition Society Conference: DICTA, pp. 309-318.

> Ferrari, R.J., Allaire, S., Hope, A., Kim, J., Jaffray, D., Pekar, V., 2011. Detection of point landmarks in 3D medical images via phase congruency model. Journal of the Brazilian Computer Society 17, 117-132.

> Villa Pinto, C.H.; Ferrari, R.J., 2016. Initialization of deformable models in 3D magnetic resonance images guided by automatically detected phase congruency point landmarks. Pattern Recognition Letters 79, 1-7.

> Ferrari, R. J.; Villa Pinto, C. H.; Moreira, C. F., 2016. Detection of the midsagittal plane in MR images using a sheetness measure from eigenanalysis of local 3D phase congruency responses. 2016 IEEE International Conference on Image Processing (ICIP), Phoenix. AZ, USA. p. 2335-2339.

...among several others. In addition, [Dr. Peter Kovesi's website](http://www.peterkovesi.com) contains some great MATLAB implementations for 2D images.

## Dependencies

- [CMake 3.6](https://cmake.org)
- [FFTW 3.3](http://www.fftw.org)
- [GSL 2.2](https://www.gnu.org/software/gsl)
- [ITK 4.10](https://www.itk.org)

## Notes

- This implementation does not make use of [monogenic filters](https://www.math.ucdavis.edu/~saito/data/phase2/monogenic.pdf) due to the application it is aimed to. So be aware that this is certainly not the fastest implementation that can be achieved.
- The NIfTI (.nii) format is used for most image outputs. You can use softwares like [3D Slicer](https://www.slicer.org) and [ITK-Snap](http://www.itksnap.org/pmwiki/pmwiki.php) to open such files.

