# The core implementation of Scalable Corpus Index (SCI) model, a method for fast retrieval on scalable corpora.

## data description
 - raw dataset:
 
   nq-train.json & nq-dev.json
 
 - passage_id_pairs: 
   Its structure is like 
   List[
      {"passage_id": ...,
       "text": ...},
      {"passage_id": ...,
       "text": ...},
       ...
    ]

    passage_id_pairs_all.json
    passage_id_pairs_train.json
    passage_id_pairs_test.json
