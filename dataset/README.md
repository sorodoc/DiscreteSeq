# DiscreteSeq

Accompanying dataset for our paper at BlackBoxNLP 2021, [Controlled tasks for model analysis: Retrieving discrete information from sequences](https://aclanthology.org/2021.blackboxnlp-1.37/).

### Contents of the folder
* It contains one directory for each task introduced in the paper.
* Every directory contains an archived version of the training data (due to size), the validation data and the test data.


### Dataset description
The trainig data files contain 100k datapoints and both the validation and the test data contain 10k datapoints.


### Datapoint stucture
Each datapoint occupies 3 lines in the files.
- The first line contains the sequence input. It starts with the word `sequence` and it is followed by n binary vectors separated by tabs.
- The second line contains the query.  It starts with the word `query` and it is followed by the sentence `have you seen x` where x represents the position id of the queried feature.
- The third line contains the anwer. It starts with the word `answer` and it is followed by either `no` or `yes`.

### Datapoint example
* sequence	000000010000000110000010110001010000	100000001001000000010000000000100000	000000000000100101000000110100000011	000010010100101100001100110000010010	000000000001000000010100000100000100
* query	have you seen 1
* answer	no



## Acknowledgements
This project has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No 715154), and from the Spanish Ram\'on y Cajal programme (grant RYC-2015-18907). We are grateful to the NVIDIA Corporation for the donation of GPUs used for this research. We are also very grateful to the Pytorch developers. This paper reflects the authors' view only, and the EU is not responsible for any use that may be made of the information it contains.
