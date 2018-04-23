# twitter-ner
Twitter NER project for Advanced Topics in LCT
This project was done in collaboration with Jovana Urosevic (@jurosevic92). 

All of our experiments can be found in the experiments.ipynb, while the results can be found in the predictions folder. The predictions folder is divided into the 10label and 2label version of the task. We mainly focused on the 10 label version of the task, and thus only two models are available for the 2 label version. The prediction folder also contains model-desc.txt which describes the configuration of each model based on the embeddings type.

Inside the 10label and 2label folder, you can find the output which contains the predictions in ConLL format [word, gold, predicted] as model-X.txt, where X is the model number and the ConLL evaluation of each model as model-X-eval.txt.
