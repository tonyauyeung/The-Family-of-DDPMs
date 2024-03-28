# The-Family-of-Denoising-Diffusion-Probabilistic-Models

This is the repo of our paper, The Family of DDPMs, @_**Cambridge-MLMI4-Advanced-Machine-Learning**_.


Authors: [Jack Naish](https://github.com/jnaish), [Tony RuiKang OuYang](https://github.com/tonyauyeung) and [Akshay Choudhry]() @ _University of Cambridge_\
Emails: &nbsp;&nbsp;{jrhn2, ro352, ac2591}@cam.ac.uk
  
## Abstract
_Denoising Diffusion Probabilistic Models_ (DDPM) are a class of generative models, inspired by non-equilibrium thermodynamics, that feature stable training regimes and are capable of producing high quality, diverse data. However, DDPMs suffer from expensive inference times compared with GANs and VAEs. Furthermore, they can't achieve competitive log-likelihoods. To solve these issues and further improve DDPMs' performace, _Denoising Diffusion Implicit Model_ (DDIM) and _Improved DDPM_ (IDDPM) have been proposed. In this paper, we systematically summarise, replicate, and compare this family of models. We show that DDPMs can generate high quality images; DDIM speed up images generation but lower sample quality; while IDDPM can speed up sampling with negligible quality reduction. Additionally, we further explore an application of DDPMs on image in-painting via _RePaint_, illustrating power of this family.

## Samples
<p align="center">
<img width="1213" alt="image" src="https://github.com/tonyauyeung/The-Family-of-DDPMs/assets/79797853/980e8424-bc4a-4d3e-9b34-4517a8e08ac1">
</p>

## Inpaint
<p align="center">
<img width="603" alt="image" src="https://github.com/tonyauyeung/The-Family-of-DDPMs/assets/79797853/73f363c3-4f7a-463f-a0cb-6284fa9183fa">
</p>

## Key References
[1] Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.\
[2] Song, J., Meng, C., and Ermon, S. Denoising Diffusion Implicit Models. arXiv, October 2020. doi: 10.48550/arXiv.2010.02502.\
[3] Nichol, A. Q. and Dhariwal, P. Improved denoising diffusion probabilistic models. In International Conference on Machine Learning, pp. 8162–8171. PMLR, 2021.\
[4] Lugmayr, A., Danelljan, M., Romero, A., Yu, F., Timofte, R., and Van Gool, L. RePaint: Inpainting using Denoising Diffusion Probabilistic Models. arXiv, January 2022. doi:10.48550/arXiv.2201.09865.

## Acknowledgement:
[1] _Machine Learning and Machine Intelligence_ @ _Cambridge_\
[2] codes reference @ [Diffusion-Models-pytorch](https://github.com/dome272/Diffusion-Models-pytorch) and [improved-diffusion](https://github.com/dome272/Diffusion-Models-pytorch)
