
<p align="center">
<img src="https://huggingface.co/spaces/anonmsr/girt-space/raw/main/assets/logo.svg" alt="GlotLID" width="30%" />
</p>
<p align="center">
<a href="https://huggingface.co/anonmsr/girt-t5-base"><img alt="HuggingFace Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-8A2BE2"></a>
<a href="https://huggingface.co/spaces/anonmsr/girt-space"><img alt="HuggingFace Demo" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space (Demo)-orange"></a>
<a href="https://huggingface.co/datasets/anonmsr/girt-instruct"><img alt="HuggingFace Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue"></a>
</p>

## TL;DR

The repository introduces **GIRT-Model**, an open-source assistant language model that automatically generates IRTs. It creates IRTs based on the developer’s
instructions regarding the structure and necessary fields.

## How to use 
```python

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained('anonmsr/girt-t5-base')
tokenizer = AutoTokenizer.from_pretrained('anonmsr/girt-t5-base')

# Ensure that the model is on the GPU for cpu use 'cpu' instead of 'cuda'
model = model.to('cuda')


# method for computing issue report template generation
def compute(sample, top_p, top_k, do_sample, max_length, min_length):

    inputs = tokenizer(sample, return_tensors="pt").to('cuda')

    outputs = model.generate(
        **inputs,
        min_length= min_length,
        max_length=max_length,
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k)

    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    generated_text = generated_texts[0]
    
    replace_dict = {
        '\n ': '\n',
        '</s>': '',
        '<pad> ': '',
        '<pad>': '',
        '<unk>!--': '<!--',
        '<unk>': '',
    }
    
    postprocess_text = generated_text
    for key, value in replace_dict.items():
        postprocess_text = postprocess_text.replace(key, value)

    return postprocess_text

prompt = "YOUR INPUT INSTRUCTION"
result = compute(prompt, top_p = 0.92, top_k=0, do_sample=True, max_length=300, min_length=30)
```
