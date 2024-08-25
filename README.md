<img align="left" width="80" height="80" alt="Shohanur Islam Sobuj" src="assets/skt_logo.png"/>

#  Parameter-Efficient Fine-Tuning of Large Language Models using Semantic Knowledge Tuning

## Abstract
In recent years, improving Large Language Models (LLMs) for specific tasks using prompts has become popular in language research due to its low computational cost. Traditional methods like prefix tuning use special, modifiable tokens that do not have real meaning, requiring a lot of training to work well and often not performing optimally. We introduce a new method called Semantic Knowledge Tuning (SK-Tuning) for prompt and prefix tuning which uses meaningful words instead of random tokens for this tuning. This method involves using a fixed LLM to first understand and process the semantic content of the prompt in zero-shot capabilities. Then, it combines the processed prompt with the input text to enhance the model's performance on specific tasks. Our experiments show that SK-Tuning trains faster, uses fewer parameters, and performs better on tasks like text classification and understanding compared to other tuning methods. This approach offers a promising way to make LLMs more efficient and effective in processing language tasks.



### Methodology
The SK-Tuning method figure is shown below:
![SK-Tuning](assets/sk_tuning.png)
> Figure 1. SK-Tuning approaches for Prefix (left) and Prompt (right). The <span style="color:red">dashed line</span> represents the optimization path during
the backward pass to the trainable adapter. Notably, in the context of prompt-tuning (on the right), the The <span style="color:red">no sign</span> signifies the
discontinuation of the forward pass beyond a certain point. This is because we exclusively initialize layer-specific semantic
information for the prompt, rendering the continuation of the forward pass unnecessary for the remaining layer



### Setup
We conduct our experiment with Anaconda3. If you have installed Anaconda3, then create the environment for P-tuning v2:

```shell
conda create -n ST python=3.10.12
conda activate ST
```

After we setup basic conda environment, install pytorch related packages via:

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Finally, install other python packages we need:

```shell
pip install -r requirements.txt
```
