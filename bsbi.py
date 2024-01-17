import os
import pickle
import contextlib
import heapq
import time
import math
import nltk

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        self.docs = {}
        self.docs_real = {}

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)
        with open(os.path.join(self.output_dir, 'docs_content.dict'), 'wb') as f:
            pickle.dump(self.docs, f)
        with open(os.path.join(self.output_dir, 'docs_content_real.dict'), 'wb') as f:
            pickle.dump(self.docs_real, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs_content.dict'), 'rb') as f:
            self.docs = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs_content_real.dict'), 'rb') as f:
            self.docs_real = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Inggris

        JANGAN LUPA BUANG STOPWORDS!

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parse_block(...).
        """
        #Stemmer
        porter = PorterStemmer()

        #Stopword Remover
        stop_words = set(stopwords.words('english'))

        #Tokenizer
        tokenizer = RegexpTokenizer(r'\w+')

        td_pairs = []
        
        directory = os.path.join(self.data_dir, block_dir_relative)
        for filename in os.listdir(directory):
            file = os.path.join(directory, filename)
            with open(file) as f:
                contents = f.read()
                self.docs_real[os.path.join(block_dir_relative, filename)] = ''.join(contents[:500].splitlines())
                word_tokens = tokenizer.tokenize(contents) #Tokenize
                output = [w for w in word_tokens if not w.lower() in stop_words] #Remove stopwords
                res = [porter.stem(w) for w in output] #Stem
                self.docs[os.path.join(block_dir_relative, filename)] = res
                for word in res:
                    td_pairs.append((self.term_id_map[word], self.doc_id_map[os.path.join(block_dir_relative, filename)]))
        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        term_dict = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = {}
            term_dict[term_id][doc_id] = term_dict[term_id].get(doc_id, 0)+1 #{doc_id : tf}
        for term_id in sorted(term_dict.keys()):
            tf_list = []
            for doc_id in term_dict[term_id].keys():
                tf_list.append(term_dict[term_id][doc_id])
            index.append(term_id, list(term_dict[term_id]), tf_list)

    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi orted_merge_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        curr, postings, tf_list = next(merged_iter) # first item
        for t, postings_, tf_list_ in merged_iter: # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) or len(self.docs) == 0:
            self.load()

        #Stemmer
        porter = PorterStemmer()

        #Stopword Remover
        stop_words = set(stopwords.words('english'))

        #Tokenizer
        tokenizer = RegexpTokenizer(r'\w+')

        word_tokens = tokenizer.tokenize(query) #Tokenize
        output = [w for w in word_tokens if not w.lower() in stop_words] #Remove stopwords
        res = [porter.stem(w) for w in output]

        #print(res)

        scores = {}
        result_topk = []
        heap = []

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory = self.output_dir) as index:
            for word in res:
                term_id = self.term_id_map[word]
                try:
                    tup = index.get_postings_list(term_id)
                    postings_list, tf_list = tup[0], tup[1]
                except KeyError:
                    postings_list, tf_list = [], []
                heapq.heappush(heap, (len(postings_list), list(zip(postings_list, tf_list))))
                while len(heap) > 1:
                    pop1 = heapq.heappop(heap)[1]
                    pop2 = heapq.heappop(heap)[1]
                    intersected = sorted_merge_posts_and_tfs(pop1, pop2)
                    heapq.heappush(heap, (len(intersected), intersected))
            
            # Count w(t, D)
            for doc_score in heapq.heappop(heap)[1]:
                if doc_score[1] > 0:
                    wtd = 1 + math.log(doc_score[1], 10)
                else:
                    wtd = 0
                # Count w(t, Q)
                try:
                    wtq = math.log(len(index.doc_length)/index.postings_dict[term_id][1], 10)
                except KeyError:
                    wtq = 0
                scores[self.doc_id_map[doc_score[0]]] = wtd*wtq

        for doc_id, score in sorted(scores.items(), key=lambda t: t[1], reverse=True):
            result_topk.append((score, doc_id))
            if len(result_topk) == k:
                break

        return result_topk

    def retrieve_bm25(self, query, k = 10):
        """
        w(t, Q) = IDF = log (N / df(t))

        w(t, D) = ((k1+1) * tf(t, D)) / k1 * ((1-b) + b * (dl / avdl)) + tf(t, D)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0 or len(self.docs) == 0:
            self.load()
            total_len = 0
            total_doc = 0
            with InvertedIndexReader(self.index_name, self.postings_encoding, directory = self.output_dir) as index:
                self.dl_dict = index.doc_length
                for doc_id, doc_len in self.dl_dict.items():
                    total_doc += 1
                    total_len += doc_len
                self.avdl = total_len / total_doc

        #Stemmer
        porter = PorterStemmer()

        #Stopword Remover
        stop_words = set(stopwords.words('english'))

        #Tokenizer
        tokenizer = RegexpTokenizer(r'\w+')

        word_tokens = tokenizer.tokenize(query) #Tokenize
        output = [w for w in word_tokens if not w.lower() in stop_words] #Remove stopwords
        res = [porter.stem(w) for w in output]

        #print(res)

        scores = {}
        result_topk = []
        heap = []

        #Modify this
        b = 0.3

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory = self.output_dir) as index:
            for word in res:
                term_id = self.term_id_map[word]
                try:
                    tup = index.get_postings_list(term_id)
                    postings_list, tf_list = tup[0], tup[1]
                except KeyError:
                    postings_list, tf_list = [], []
                heapq.heappush(heap, (len(postings_list), list(zip(postings_list, tf_list))))
                while len(heap) > 1:
                    pop1 = heapq.heappop(heap)[1]
                    pop2 = heapq.heappop(heap)[1]
                    intersected = sorted_merge_posts_and_tfs(pop1, pop2)
                    heapq.heappush(heap, (len(intersected), intersected))
            
            # Count w(t, D)
            for doc_score in heapq.heappop(heap)[1]:
                wtd = ((k + 1) * doc_score[1]) / (k * ((1-b) + b * (self.dl_dict[doc_score[0]] / self.avdl)) + doc_score[1])
                # Count w(t, Q)
                try:
                    wtq = math.log(len(index.doc_length)/index.postings_dict[term_id][1], 10)
                except KeyError:
                    wtq = 0
                scores[self.doc_id_map[doc_score[0]]] = wtd*wtq

        for doc_id, score in sorted(scores.items(), key=lambda t: t[1], reverse=True):
            result_topk.append((score, doc_id))
            if len(result_topk) == 100:
                break

        return result_topk

    def index(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None
    
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                              postings_encoding = VBEPostings, \
                              output_dir = 'index')
    BSBI_instance.index() # memulai indexing!
