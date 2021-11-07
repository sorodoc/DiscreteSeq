# DiscreteSeq

Accompanying dataset and code for our paper at BlackBoxNLP 2021, [Controlled tasks for model analysis: Retrieving discrete information from sequences](https://aclanthology.org/2021.blackboxnlp-1.37/).

##### Citation

```
@inproceedings{sorodoc-etal-2021-controlled,
    title = "Controlled tasks for model analysis: Retrieving discrete information from sequences",
    author = "Sorodoc, Ionut-Teodor  and
      Boleda, Gemma  and
      Baroni, Marco",
    booktitle = "Proceedings of the Fourth BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.blackboxnlp-1.37",
    pages = "468--478",
}
```

### Contents of this repository
* The datasets associated with the seven new tasks introduced in the paper in the folder `dataset`
* The basic models used for our experiments in the folder `models`


### Usage of main.py for training models
Run the `main.py` script with the corresponding parameters. 

Some important parameters are: 
- `model` can be `Transformer` or `LSTM`
- `q_type` specifies the Task that the model will learn. 
- `self_attn` specifies if the Transformer contains self attention. 
- `seq_size` sets the length of the input sequences (between 5, 10, 15, 20, 25, 30).
- `query_attn` specifies if the model includes decoder attention between the query and the input sequence.
- `pos_encoding` specifies if Transformer includes positional encoding.
- `no_cuda` is set to run the system in CPU mode.

#### Example: Training the complete Transformer model on the first task with sequences containing 10 elements.
`python main.py --model Transfoermer --q_type Task_1 --self_attn 1 --seq_size 10 --query_attn 1 --pos_encoding 1`


## Acknowledgements
This project has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No 715154), and from the Spanish Ram\'on y Cajal programme (grant RYC-2015-18907). We are grateful to the NVIDIA Corporation for the donation of GPUs used for this research. We are also very grateful to the Pytorch developers. This paper reflects the authors' view only, and the EU is not responsible for any use that may be made of the information it contains.
