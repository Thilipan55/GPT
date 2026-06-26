# Generative (Only) Pretrained Transformer

21/June
This repository has a scaled dowedn representation of GPT-2 by OpenAI and structurally is similar to the Transformer block in Attention is all you need (Vaswani et al., 2017), This is a really small version of that and trianed on roughly 1-Million tokens on shakespearean text

This is a slightly modified replication of Andrej Karpathys NanoGPT from his Series : https://github.com/karpathy/nanogpt

To learn more about the performance on this tiny model, kindly visit my blog which has well somewhat detailed evals on every step in how this model was built
Blog -> https://scandalous-result-48a.notion.site/GPT-architecture-from-scratch-35f6add4fddd80428f4be2d128166ad9

This is a very samll model and it cant compess 1 million tokens into the model parameter, speaking of tokenizer this is just a dictionary of all the voacbulary, the letters around 65 with letters lower, upper, and some special tokens

The final goal is to scale up this model and train it on the Tamil literature: Thirukkural https://github.com/b1zantine/thirukkural-dataset/blob/master/thirukkural.txt
This should be intersting as Thirukkural implicitly has strict syntax with 4 words on top and 3 in below

---

Log - 22/June
I Scaled the parameters up from 50,000 to 930K parameters so almost a million, i trained it on a T4 instance in Colab and achved a loss -> 1.15025
Blog for more information -> https://scandalous-result-48a.notion.site/Scaling-GPT-2-Transformer-1-Million-3876add4fddd80c7b157c7facb08f31e

Now you can just
```bash
git clone https://github.com/Thilipan55/GPT.git
cd GPT
cd GPT-s1-scaled
```

```python
model = Bigram(vocab_size).to(device)

check = torch.load("checkpoint_80000.pt")
model.load_state_dict(check["model_weights"])

# and run infernce with
print(decode((old_model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), 500))[0].tolist()))
```
And run the model :)
---
Log - 25/June
I am releasing Kural-1-mini today, This is an autoregressive decoder only transformer (same architecture as previous) and, it was only pretrained on thirukkural couplets, and the tokenizer is still (unfortunatley) only character level.

Blog on Kural-1-mini and Kural-1-High -> https://scandalous-result-48a.notion.site/Kural-1-Family-of-models-3886add4fddd806b9dd7fb9b9752b08a

The weights are also present in the repsository in /kural-1 and run the same command above to load the model weights and run inference.

Log - 26/Jun
I am releasing Kural-1-high this is a scaled version of the same model as above
I lost the weights so only code is there :( ,thanks a lot colab runtime
