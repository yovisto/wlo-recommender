# WLO Recommender

A [Docker](https://docker.com/)/[Python](https://www.python.org/)/[Keras](https://keras.io/)/[Tensorflow](https://www.tensorflow.org/) utility implementing a recommender tool based on document embeddings for the [WLO project](https://github.com/openeduhub/) dataset.

 
## Prerequisites

- Install [Docker](https://docker.com/).
- (The training script `runTraining.sh` requires the Nvidia Docker runtime installed. For processing without a GPU remove the `--runtime=nvidia` parameter in the script's docker command.)

- Build the Docker container.

```
sh build.sh
```

## Training

(The `data` folder already containes a pretrained model.)

- The following script retrieves and processes the latest [dataset](https://github.com/openeduhub/oeh-wlo-data-dump), which results in the `data/wirlernenonline.oeh.csv` file containing the relevant documents (documents with a discipline property).

```
sh prepareData.sh
```

- This script initiates the training, which results in the model file `data/wirlernenonline.oeh-embed.h5`, and a serialization of the document ids list `data/wirlernenonline.oeh-id.pickle` (existing files will be overwritten without warning).

```
sh runTraining.sh
```

## Prediction

- To test the prediction just query the model with an arbitrary document id.

```
sh runPrediction.sh 4d8aa27e-b102-417b-8f52-fe9e57620308
```
- The result is a list of scores and document ids relevant to the query document. Only the top ten items are retrieved, in descending order.

## Webservice

- To run the recommender tool as a simple REST based webservice, the following script can be used:

```
sh runService.sh
```

- The scripts deploys a CherryPy webservice in a docker container listening at `http://localhost:8080/recommend`.

- To retrieve the recommendations, create a POST request and submit a json document with a document id as for example: 

```
curl -d '{"doc" : "1e6e21fd-e5d6-4046-a532-803708ea130d"}' -H "Content-Type: application/json" -X POST http://0.0.0.0:8080/recommend
```	
