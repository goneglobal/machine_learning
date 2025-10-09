---
base_model: HuggingFaceH4/tiny-random-LlamaForCausalLM
library_name: transformers
model_name: unsloth_cpu_toy
tags:
- generated_from_trainer
- sft
- trl
licence: license
---

# Model Card for unsloth_cpu_toy

This model is a fine-tuned version of [HuggingFaceH4/tiny-random-LlamaForCausalLM](https://huggingface.co/HuggingFaceH4/tiny-random-LlamaForCausalLM).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with SFT.

### Framework versions

- TRL: 0.23.1
- Transformers: 4.56.2
- Pytorch: 2.2.2
- Datasets: 4.1.1
- Tokenizers: 0.22.1

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```