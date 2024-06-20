<div align="center">
  <img src="./assets/Volcano-Seashore.jpg" alt="" width="100%">

  <br>

  # <img src="./assets/icons/EvoSeed.png" style="height:50px;"> **EvoSeed** <img src="./assets/icons/EvoSeed.png" alt="EvoSeed" class="image" style="height:50px;">

  <br>

  <img src="./assets/cover.jpg" alt="Publication" width="100%">

</div>

<img src="./assets/icons/publication.png" style="height:20px;"> Source for the article: [Breaking Free: How to Hack Safety Guardrails in Black-Box Diffusion Models!](https://arxiv.org/abs/2402.04699)

> <img src="./assets/icons/firework.png" style="height:20px;"> (New!) Added Tutorial to generate adversarial images for ResNet-50 using Stable Diffusion.


## <img src="./assets/icons/contributions.png" style="height:35px;"> Key Contributions:

- We propose a black-box algorithmic framework based on an Evolutionary Strategy titled EvoSeed to generate natural adversarial samples in an unrestricted setting.
- Our results show that adversarial samples created using EvoSeed are photo-realistic and do not change the human perception of the generated image; however, can be misclassified by various robust and non-robust classifiers.

<div align="center">
    <img src="./assets/demo.jpg" alt="" width="100%">
</div>
Figure: Adversarial images created with EvoSeed are prime examples of how to deceive a range of classifiers tailored for various tasks.
Note that, the generated natural adversarial images differ from non-adversarial ones, suggesting the adversarial images' unrestricted nature.

## <img src="./assets/icons/mortarboard.png" style="height:35px"> Tutorial:
Tutorial for creating adversarial images for ResNet-50 using Stable Diffusion can be found in the [notebook](./code/Tutorial.ipynb)

<div align="center">
    <img src="./assets/output.gif" alt="" width="100%">
</div>

## <img src="./assets/icons/cite.png" style="height:35px"> Citation:

If you find this project useful please cite:

```bibtex
@article{kotyan2024EvoSeed,
  title = {Breaking Free: How to Hack Safety Guardrails in Black-Box Diffusion Models!,
  author = {Kotyan, Shashank and Mao, Po-Yuan and Chen, Pin-Yu and Vargas, Danilo Vasconcellos},
  year = {2024},
  month = may,
  number = {arXiv:2402.04699},
  eprint = {2402.04699},
  publisher = {{arXiv}},
  doi = {10.48550/arXiv.2402.04699},
}
```
