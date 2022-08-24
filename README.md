# Relation-Classification-via-Convolutional-Deep-Neural-Network
the code for the paper “Relation Classification via Convolutional Deep Neural Network”

## Environment Requirements
* python 3.9
* pytorch 1.3.0

## Data
* [SemEval2010 Task8](https://github.com/CrazilyCode/SemEval2010-Task8) [[paper](https://www.aclweb.org/anthology/S10-1006.pdf)]
* [Embedding - Turian et al.(2010)](http://metaoptimize.s3.amazonaws.com/hlbl-embeddings-ACL2010/hlbl-embeddings-scaled.EMBEDDING_SIZE=50.txt.gz) [[paper](https://www.aclweb.org/anthology/P10-1040.pdf)\]


## Usage
1. Download the embedding and decompress it into the `embedding` folder.
2. Run the following the commands to start the program.
```shell
python run.py
```
More details can be seen by `python run.py -h`.

3. You can use the official scorer to check the final predicted result.
```shell
perl semeval2010_task8_scorer-v1.2.pl proposed_answer.txt predicted_result.txt >> result.txt
```

## Result
The result of my version and that in paper are present as follows:
| paper | my version |
| :------: | :------: |
| 0.789 | 0.7876 |

The training log can be seen in `train.log` and the official evaluation results is available in `result.txt`.

