from bsbi import BSBIIndex
from compression import VBEPostings
from letor import LETOR

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

LETOR_instance = LETOR()

queries = ["alkylated with radioactive iodoacetate", \
           "psychodrama for disturbed children", \
           "lipid metabolism in toxemia and normal pregnancy"]

for query in queries:
    docs = []
    for (score, doc) in BSBI_instance.retrieve_bm25(query, k = 10):
        docs.append((doc, " ".join(BSBI_instance.docs[doc]), BSBI_instance.docs_real[doc]))
 
    sorted_did_scores = LETOR_instance.predict(query, docs)
    print("Query  : ", query)
    print("Results:")
    for (did, score, content) in sorted_did_scores:
        print(f"{did:30} {score:>.2f} {content}")
    print()