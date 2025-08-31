# Finetuning a LLM 🚀

# The Project 🚀

This is a fun project that came into existence after my friends and I came up with the idea of finetuning an LLM for a seminar about Artificial Intelligence. We are three people with none of us having any prior experience in this field.

The idea was to take a pretrained LLM, of a size small enough so the training process can run on our home computers, and finetune it with a dataset to give it human-like behavior.

Below are links to the Model and Dataset we used:

-Model: [https://huggingface.co/microsoft/phi-2](https://huggingface.co/microsoft/phi-2)

-Dataset: [https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset](https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset)

# The Paper 📝

We wrote down the whole journey as well as findings, results, methods, math, and analytics of this project in the following Paper:

[Feinabstimmung_eines_LLMs_für_menschenähnliche_Textgenerierung.pdf](https://github.com/lacedicetea/KI-project/edit/main/Feinabstimmung_eines_LLMs_für_menschenähnliche_Textgenerierung.pdf)

Since we wrote it in LaTeX, I cannot embed it into this README directly.
Furthermore, it is written in German since we attend a German university.

For the finetuning itself, we used QLoRA to efficiently reduce the computing needed for the finetuning, allowing us to use a bigger model. More on that in our paper.

# License 📝

This project is licensed under the MIT License.

#

Created with ❤️ by lacedicetea & co
