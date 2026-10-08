# Density functional perturbation theory of meta-generalized gradient approximations using algorithmic differentiation

Code and data supporting the paper:
> Bruno Ploumhans, Niklas Frederik Schmitz, Michael F. Herbst.
> Density functional perturbation theory of meta-generalized gradient approximations using algorithmic differentiation. [arXiv:2609.35572 [cond-mat.mtrl-sci]](https://arxiv.org/abs/2609.35572)

If you use this code, please cite:
```bibtex
@misc{ploumhans2026mgga_addfpt,
      title={Density functional perturbation theory of meta-generalized gradient approximations using algorithmic differentiation}, 
      author={Bruno Ploumhans and Niklas Frederik Schmitz and Michael F. Herbst},
      year={2026},
      eprint={2609.35572},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2609.35572}, 
}
```

## Repository contents
- `DFTK-mgga/`: Modified version of DFTK v0.7.25 with its DFPT implementation extended to support meta-GGAs. Some of the modifications are being or have been integrated in later releases of DFTK.
- `scripts/`: Scripts used to obtain the results and figures shown in the paper, see [the corresponding README](scripts/README.md) for a detailed explanation.
- `metapsp-patch.diff`: Our patches to METAPSP on top of the alpha 1.0.1 release from http://www.mat-simresearch.com/

## License
All code in the `scripts/` folder is licensed under the MIT license.

> The MIT License (MIT)
>
> Copyright (c) 2026 Bruno Ploumhans and collaborators
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

All data in the `scripts/` folder is licensed under [the **Creative Commons Attribution 4.0 (CC-BY 4.0)** license](https://creativecommons.org/licenses/by/4.0/).

The contents of the `DFTK-mgga/` folder are licensed under the [DFTK MIT LICENSE](DFTK-mgga/LICENSE).

The `metapsp-patch.diff` file is licensed under [the **Creative Commons Zero v1.0 Universal (CC0 1.0)** license](https://creativecommons.org/publicdomain/zero/1.0/).

## Funding
This research was supported by the Swiss National Science Foundation (SNSF, Grant No. 10002757) as well as the NCCR MARVEL, a National Centre of Competence in Research, funded by the SNSF (Grant No. 205602).