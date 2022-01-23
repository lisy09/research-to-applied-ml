# Data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language

> Data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language ([Paper](https://scontent-nrt1-1.xx.fbcdn.net/v/t39.8562-6/271974914_483120576492438_4239522333319653600_n.pdf?_nc_cat=107&ccb=1-5&_nc_sid=ae5e01&_nc_ohc=HLSTIdOnYI4AX-9U2Q6&_nc_ht=scontent-nrt1-1.xx&oh=00_AT96s23qbFCIMVMxjONyqnNePWcxE18GiKpzwhpatA15xQ&oe=61F1FD91), [Code](https://github.com/pytorch/fairseq/tree/main/examples/data2vec))

## Paper Reading

Present `data2vec`, a framework that uses the same learning method for either speech, NLP or computer vision.
The core idea is to `predict latent representations` of the full input data based on a `masked view` of the input in a self-distillation setup using a standard Transformer architecture. 

