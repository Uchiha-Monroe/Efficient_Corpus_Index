# The core implementation of ECI (an Efficient Corpus Indexer for dynamic corpora Retrieval).

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
