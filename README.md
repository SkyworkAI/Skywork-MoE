
<!-- <div align="center">
<h1>
  ✨Skywork
</h1>
</div> -->
<div align="center"><img src="misc/skywork_logo.jpeg" width="550"/></div>

<p align="center">
🤗 <a href="https://huggingface.co/Skywork" target="_blank">Hugging Face</a> • 🤖 <a href="https://modelscope.cn/organization/Skywork" target="_blank">ModelScope</a> • 👾 <a href="https://wisemodel.cn/organization/Skywork" target="_blank">Wisemodel</a> • 💬 <a href="https://github.com/SkyworkAI/Skywork/blob/main/misc/wechat.png?raw=true" target="_blank">WeChat</a>• 📜<a href="http://arxiv.org/abs/2310.19341" target="_blank">Tech Report</a>
</p>

<div align="center">

[![GitHub Stars](https://img.shields.io/github/stars/SkyworkAI/Skywork-MoE)](https://github.com/SkyworkAI/Skywork-MoE/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/SkyworkAI/Skywork-MoE)](https://github.com/SkyworkAI/Skywork-MoE/fork)
</div>

<div align="center">

</div>


# Project Introduction

Skywork-MoE is a high-performance mixture-of-experts (MoE) model with 146 billion parameters, 16 experts, and 22 billion activated parameters. This model is initialized from the pre-existing dense checkpoints of our Skywork-13B model.

We introduce two innovative techniques: Gating Logit Normalization, which enhances expert diversification, and Adaptive Auxiliary Loss Coefficients, which allow for layer-specific adjustment of auxiliary loss coefficients.

Skywork-MoE demonstrates comparable or superior performance to models with more parameters or more activated parameters, such as Grok-1, DBRX, Mistral 8*22, and Deepseek-V2.

# News and Updates
* 2024.6.3  We release the **Skywork-MoE-base** model.

# Table of contents

- [☁️Download URL](#Download-URL)
- [👨‍💻Model Introduction](#Model-Introduction)
- [🏆Model Evaluation](#Model-Evaluation)
- [⚠️Declaration and License Agreement](#Declaration-and-License-Agreement)
- [🤝Contact Us and Citation](#Contact-Us-and-Citation)


# Download URL

|         | HuggingFace Model   |  ModelScope Model   |  Wisemodel Model  |
|:-------:|:-----------:|:-----------------------------:|:-----------------------------:|
| **Skywork-MoE-base**      | 🤗 [Skywork-MoE-base](https://huggingface.co/Skywork/Skywork-MoE-base)  | 🤖[Skywork-MoE-base](https://www.modelscope.cn/models/skywork/Skywork-MoE-base) | 👾[Skywork-MoE-base](https://wisemodel.cn/models/Skywork/Skywork-MoE-base) |


# Benchmark Results
We evaluated Skywork-MoE-base model on various popular benchmarks, including C-Eval, MMLU, CMMLU, GSM8K, MATH and HumanEval.
<img src="misc/skywork_moe_base_evaluation.png" alt="Image" width="600" height="280">

# Declaration and License Agreement


## Declaration

We hereby declare that the Skywork model should not be used for any activities that pose a threat to national or societal security or engage in unlawful actions. Additionally, we request users not to deploy the Skywork model for internet services without appropriate security reviews and records. We hope that all users will adhere to this principle to ensure that technological advancements occur in a regulated and lawful environment.

We have done our utmost to ensure the compliance of the data used during the model's training process. However, despite our extensive efforts, due to the complexity of the model and data, there may still be unpredictable risks and issues. Therefore, if any problems arise as a result of using the Skywork open-source model, including but not limited to data security issues, public opinion risks, or any risks and problems arising from the model being misled, abused, disseminated, or improperly utilized, we will not assume any responsibility.

## License Agreement

The community usage of Skywork model requires [Skywork Community License](https://github.com/SkyworkAI/Skywork-MoE/blob/main/Skywork%20Community%20License.pdf). The Skywork model supports commercial use. If you plan to use the Skywork model or its derivatives for commercial purposes, you must abide by terms and conditions within [Skywork Community License](https://github.com/SkyworkAI/Skywork-MoE/blob/main/Skywork%20Community%20License.pdf).

  

[《Skywork 模型社区许可协议》》]:https://github.com/SkyworkAI/Skywork-MoE/blob/main/Skywork%20模型社区许可协议.pdf


[skywork-opensource@kunlun-inc.com]: mailto:skywork-opensource@kunlun-inc.com

# Contact Us and Citation
If you find our work helpful, please feel free to cite our paper~
```
@misc{wei2023skywork,
      title={Skywork: A More Open Bilingual Foundation Model}, 
      author={Tianwen Wei and Liang Zhao and Lichang Zhang and Bo Zhu and Lijie Wang and Haihua Yang and Biye Li and Cheng Cheng and Weiwei Lü and Rui Hu and Chenxia Li and Liu Yang and Xilin Luo and Xuejie Wu and Lunan Liu and Wenjun Cheng and Peng Cheng and Jianhao Zhang and Xiaoyu Zhang and Lei Lin and Xiaokun Wang and Yutuan Ma and Chuanhai Dong and Yanqi Sun and Yifu Chen and Yongyi Peng and Xiaojuan Liang and Shuicheng Yan and Han Fang and Yahui Zhou},
      year={2023},
      eprint={2310.19341},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


