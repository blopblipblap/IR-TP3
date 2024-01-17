from cmath import cos
import random
import lightgbm
import numpy as np
import pickle
import os

from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine

NUM_NEGATIVES = 1
NUM_LATENT_TOPICS = 200

class LETOR:
    def __init__(self):
        self.documents = {}
        self.queries = {}

        with open("nfcorpus/train.docs") as file:
            for line in file:
                doc_id, content = line.split("\t")
                self.documents[doc_id] = content.split()

        with open("nfcorpus/train.vid-desc.queries", encoding='utf-8') as file:
            for line in file:
                q_id, content = line.split("\t")
                self.queries[q_id] = content.split()
        
    def training_prep(self):
        #relevance level: 3 (fullt relevant), 2 (partially relevant), 1 (marginally relevant)
        #
        #grouping by q_id   
        self.q_docs_rel = {}
        with open("nfcorpus/train.3-2-1.qrel") as file:
            for line in file:
                q_id, _, doc_id, rel = line.split("\t")
                if (q_id in self.queries) and (doc_id in self.documents):
                    if q_id not in self.q_docs_rel:
                        self.q_docs_rel[q_id] = []
                    self.q_docs_rel[q_id].append((doc_id, int(rel)))

        #group_qid_count untuk model LGBMRanker
        self.group_qid_count = []
        self.dataset = []
        for q_id in self.q_docs_rel:
            docs_rels = self.q_docs_rel[q_id]
            self.group_qid_count.append(len(docs_rels) + NUM_NEGATIVES)
            for doc_id, rel in docs_rels:
                self.dataset.append((self.queries[q_id], self.documents[doc_id], rel))
            #tambahkan satu negative
            self.dataset.append((self.queries[q_id], random.choice(list(self.documents.values())), 0))
        

    def building_model(self):
        self.dictionary = Dictionary()
        bow_corpus = [self.dictionary.doc2bow(doc, allow_update = True) for doc in self.documents.values()]
        self.model = LsiModel(bow_corpus, num_topics = NUM_LATENT_TOPICS) # 200 latent topics
        with open(os.path.join('model', 'lsi_model.sav'), 'wb') as f:
            pickle.dump(self.model, f)
        with open(os.path.join('model', 'bow_corpus.dict'), 'wb') as f:
            pickle.dump(self.dictionary, f)

    def vector_rep(self, text):
        if hasattr(self, 'model') == False:
            with open(os.path.join('model', 'lsi_model.sav'), 'rb') as f:
                self.model = pickle.load(f)
            with open(os.path.join('model', 'bow_corpus.dict'), 'rb') as f:
                self.dictionary = pickle.load(f)
        rep = [topic_value for (_, topic_value) in self.model[self.dictionary.doc2bow(text)]]
        return rep if len(rep) == NUM_LATENT_TOPICS else [0.] * NUM_LATENT_TOPICS

    def features(self, query, doc):
        v_q = self.vector_rep(query)
        v_d = self.vector_rep(doc)
        q = set(query)
        d = set(doc)
        cosine_dist = cosine(v_q, v_d)
        jaccard = len(q & d) / len(q | d)
        return v_q + v_d + [jaccard] + [cosine_dist]
    
    def separate_dataset(self):
        self.X = []
        self.Y = []
        for (query, doc, rel) in self.dataset:
            self.X.append(self.features(query, doc))
            self.Y.append(rel)
        
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)

    def training_ranker(self):
        self.ranker = lightgbm.LGBMRanker(
                            objective="lambdarank",
                            boosting_type = "gbdt",
                            n_estimators = 100,
                            importance_type = "gain",
                            metric = "ndcg",
                            num_leaves = 40,
                            learning_rate = 0.02,
                            max_depth = -1)

        self.ranker.fit(self.X, self.Y,
                group = self.group_qid_count,
                verbose = 10)
        
        with open(os.path.join('model', 'lightgbm_model.sav'), 'wb') as f:
            pickle.dump(self.ranker, f)

    def training(self):
        #training preparations
        self.training_prep()

        #building LSI/LSA Model
        self.building_model()
        self.separate_dataset()

        #training the ranker
        self.training_ranker()

    def predict(self, query, docs):
        with open(os.path.join('model', 'lightgbm_model.sav'), 'rb') as f:
            self.ranker = pickle.load(f)

        X_unseen = []

        for doc_id, doc, doc_real in docs:
            X_unseen.append(self.features(query.split(), doc.split()))
        X_unseen = np.array(X_unseen)
        scores = self.ranker.predict(X_unseen)

        did_scores = [x for x in zip([did for (did, _, _) in docs], scores, [doc for (_, _, doc) in docs])]
        sorted_did_scores = sorted(did_scores, key = lambda tup: tup[1], reverse = True)

        return sorted_did_scores

if __name__ == '__main__':

    LETOR_instance = LETOR()
    LETOR_instance.training()


    




