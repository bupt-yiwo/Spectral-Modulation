# Spectral Modulation

This repository contains code for the paper _"Unleashing the Potential of Large Language Models through Spectral Modulation"_ by Peng Sun and Yao Zhu [EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.224.pdf). 

![image-20241113213356986](C:\Users\yiwo\AppData\Roaming\Typora\typora-user-images\image-20241113213356986.png)

We welcome issues and pull requests!

## Core Algorithm Code

```python
import torch.fft as fft


def Fourier_filter_low(x, threshold, scale):
    dtype = x.dtype
    H, W = x.shape
    # Non-power of 2 images must be float32
    if (W & (W - 1)) != 0 or (H & (H - 1)) != 0:
        x = x.type(torch.float32)
    # FFT
    x_freq = fft.fftn(x)
    x_freq = fft.fftshift(x_freq)
    
    H, W = x_freq.shape
    mask = scale*torch.ones((H, W)).to(x.device) 
    # print(H,W)
    crow, ccol = H // 2, W //2
    mask[crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = 1
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq)
    x_filtered = fft.ifftn(x_freq).real
    
    x_filtered = x_filtered.type(dtype)
    return x_filtered


def do_low_pass(weight, k,threshold1 , debug=False, niter=20):
    assert weight.ndim == 2
    weight_approx = Fourier_filter_low(weight, threshold= threshold1, scale=k)
    weight_approx = weight_approx - weight_approx.mean() + weight.mean()
    print("Low Frequency:", weight_approx.max(), weight_approx.min(),weight_approx.mean(),weight_approx.std())
    print("Weight:", weight.max(), weight.min(),weight.mean(),weight.std())
    print("DIFF:",torch.sqrt(torch.sum((weight-weight_approx)*(weight-weight_approx))))
    
    weight_approx = torch.nn.Parameter(weight_approx)
    return weight_approx

def do_high_pass(weight, k,threshold1 , debug=False, niter=4):
    assert weight.ndim == 2
    
    temp = weight
    weight_approx = Fourier_filter_low(weight, threshold= threshold1, scale=k)
    weight_approx = weight_approx - weight_approx.mean() + weight.mean()
    print("Low Frequency:", weight_approx.max(), weight_approx.min(),weight_approx.mean(),weight_approx.std())
    print("Weight:", weight.max(), weight.min(),weight.mean(),weight.std())
    print("DIFF:",torch.sqrt(torch.sum((weight-weight_approx)*(weight-weight_approx))))
    weight_approx = temp - weight_approx

    weight_approx = torch.nn.Parameter(weight_approx)
    return weight_approx
```

## How to run 

### Datasets

If you want to experiment with the CounterFact dataset then run the following script to download it. All other datasets are available on HuggingFace.

```bash
python scripts/get_counterfact.py
```

### Run

We take the running of the Vicuna-7B-V1.5 model on the Fever dataset as an example, you can run:

```python
python vicuna_fever.py \
	--intervention do_low_pass \
    --threshold 64 \
	--rate 0.95 \
    --lnum 30 \
    --lname k_proj \
    --model_path "lmsys/vicuna-7b-v1.5" \
    --llm_name "vicuna-7b-v1.5" \
    --home_dir "./results/Fever/vicuna_7b_results" \
```

Here, threshold is the protection threshold, rate is the reduction factor, lnum is the layer you want to adjust, and lname is the weight matrix you want to modify. The mapping for _lname_ that we use is:

| **lname** | **description**                                              |
| --------- | ------------------------------------------------------------ |
| dont      | use the base model and dont perform intervention             |
| fc_in     | first layer of MLP                                           |
| fc_out    | second layer of MLP                                          |
| fc_up     | a third MLP weight matrix in some LLM, used for Hadamard multiplication |
| mlp       | all MLP weight matrices {fc_in, fc_up, fc_out}               |
| k_proj    | key matrix in self attention                                 |
| v_proj    | value matrix in self attention                               |
| q_proj    | query matrix in self attention                               |
| out_proj  | output matrix in self attention                              |
| attn      | all attention weight matrices                                |

## Citation

If you find this codebase useful, then please cite the following paper. Additionally, feel free to send a PR or an email and we will cite your result/paper on the leaderboard.

```bash
@inproceedings{sun-etal-2024-unleashing,
    title = "Unleashing the Potential of Large Language Models through Spectral Modulation",
    author = "Sun, Peng  and
      Zhu, Yao  and
      Zhang, Yunjian  and
      Yan, Xiu  and
      Wang, Zizhe  and
      Ji, Xiangyang",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.224",
    pages = "3892--3911",
    abstract = "Large Language Models (LLMs) have demonstrated impressive capabilities across various domains, garnering significant attention from both academia and industry. However, enhancing the performance of LLMs typically requires scaling up model sizes or fine-tuning with additional datasets, which results in substantial computational costs. This paper poses an intriguing question: Can we improve the performance of LLMs without additional training? Drawing inspiration from signal processing principles, which suggest that noise often resides in high-frequency components while low-frequency components carry the essence of signals, we propose uncovering untapped potential in LLMs from a frequency perspective. We hypothesize that the high-frequency components in the weight matrices of LLMs{'} linear layers may conceal noise that interferes with predictive accuracy. Therefore, we propose conducting spectral modulation in the parameter space of LLMs, which can seamlessly integrate with various models in a plug-and-play manner. Extensive experiments have demonstrated the superiority of our approach, with spectral modulation yielding an average performance improvement of up to 10.12{\%}.",
}
```
