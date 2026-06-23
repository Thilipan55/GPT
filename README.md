# Generative (Only) Pretrained Transformer

21/June
This repository has a scaled dowedn representation of GPT-2 by OpenAI and structurally is similar to the Transformer block in Attention is all you need (Vaswani et al., 2017), This is a really small version of that and trianed on roughly 1-Million tokens on shakespearean text

This is a slightly modified replication of Andrej Karpathys NanoGPT from his Series : https://github.com/karpathy/nanogpt

To learn more about the performance on this tiny model, kindly visit my blog which has well somewhat detailed evals on every step in how this model was built
Blog -> https://scandalous-result-48a.notion.site/GPT-architecture-from-scratch-35f6add4fddd80428f4be2d128166ad9

This is a very samll model and it cant compess 1 million tokens into the model parameter, speaking of tokenizer this is just a dictionary of all the voacbulary, the letters around 65 with letters lower, upper, and some special tokens

The final goal is to scale up this model and train it on the Tamil literature: Thirukkural https://github.com/b1zantine/thirukkural-dataset/blob/master/thirukkural.txt
This should be intersting as Thirukkural implicitly has strict syntax with 4 words on top and 3 in below


Log - 22/June
I Scaled the parameters up from 50,000 to 930K parameters so almost a million, i trained it on a T4 instance in Colab and achved a loss -> 1.15025

Now you can just
```bash
git clone https://github.com/Thilipan55/GPT.git
cd GPT
```

```python
model = Bigram(vocab_size).to(device)

check = torch.load("checkpoint_80000.pt")
model.load_state_dict(check["model_weights"])

# and run infernce with
print(decode((old_model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), 500))[0].tolist()))
```
And run the model :)
