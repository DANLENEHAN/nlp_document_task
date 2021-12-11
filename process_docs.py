"""
Name: Daniel Lenehan

Email: daniel.lenehan.work@gmail.com

Phone number: 07380831531
"""

from collections import Counter
import glob
from typing import Dict
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk import ngrams
from pandas.core.frame import DataFrame

from sklearn.feature_extraction.text import TfidfVectorizer


def setup_script() -> None:
    """
    Make sure the script runs smoothly
    without exceptions
    """

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')


def extract_documents(
    file_paths: list[str]
) -> DataFrame:
    """
    Takes a list of file paths as an arg.
    Returns a pandas dataframe with the raw
    documents and doc name as columns

    When opening the file below we must specify
    an encoding. If we leave this none it'll use
    locale.getpreferredencoding() which might have
    unexpected results. In my case for windows,
    this is 'cp1252'. So it will error as
    0x9d isn't defined in cp1252. It's a
    'RIGHT DOUBLE QUOTATION MARK'
    """

    documents_dict = {
        "doc_name": [],
        "raw_doc": []
    }
    for file_path in file_paths:
        f = open(file_path, "r", encoding="utf8")
        document = f.read()

        doc_name = file_path.split("\\")[1].split(".")[0]
        documents_dict['raw_doc'].append(document)
        documents_dict['doc_name'].append(doc_name)

        f.close()

    return pd.DataFrame(documents_dict)


def tokenize_document_words(
    docs_pdf: DataFrame
) -> None:
    """
    Splitting each of the raw document
    strings into words and adding the
    the result to a column in the pandas
    dataframe provided as an arg
    """

    tokenizer = RegexpTokenizer(r'\w+')
    doc_words = [
        tokenizer.tokenize(
            document
        ) for document in docs_pdf.raw_doc
    ]
    docs_pdf['doc_words'] = doc_words


def split_docs_to_sentences(
    docs_pdf: DataFrame
) -> None:
    """
    Takes a documents dataframe as an arg.
    Splits each of the raw document strings
    into sentences. Also removes any newline
    characters.
    """

    get_doc_sentences = lambda document: [
        sentence.replace("\n", "") for sentence in document.split(".")
    ]
    docs_pdf['sentences'] = docs_pdf.raw_doc.apply(get_doc_sentences)


def clean_and_preprocess_docs(
    docs_pdf: DataFrame
) -> None:
    """
    Takes the document dataframe as an arg.
    Preprocesses each of the tokenized columns in the
    document pdf.
    Pre-processing steps include decaptilizing all words
    and removing any stopwords.
    """

    decap_words = lambda document: [
        word.lower() for word in document
    ]
    docs_pdf.doc_words = docs_pdf.doc_words.apply(decap_words)

    stop_words = stopwords.words('english')
    remove_stop_words = lambda document: [
        word for word in document if word not in stop_words
    ]
    docs_pdf.doc_words = docs_pdf.doc_words.apply(remove_stop_words)


def advanced_preprocessing(
    docs_pdf: DataFrame
) -> None:
    """
    Stemming and Lemmatization are useful
    to lock into the subject of a
    document. If we can reduce words to their
    base it means we have a better chance of seeing
    patterns then if we treated each variation of the
    same word as a different word.
    Didn't end up using this one as some of the lemming
    and stemming produces poor results.
    """

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    lemmatize_words = lambda document: [
        lemmatizer.lemmatize(word) for word in document
    ]
    stem_words = lambda document: [
        stemmer.stem(word) for word in document
    ]
    docs_pdf.doc_words = docs_pdf.doc_words.apply(lemmatize_words)
    docs_pdf.doc_words = docs_pdf.doc_words.apply(stem_words)


def generate_word_summary_data(
    docs_pdf: DataFrame
) -> None:
    """
    Takes the document dataframe as an
    arg and summaries of each of the documents.
    The summary includes:
        - Occurrence count of each word for each doc
        - Generates all the bigrams and trigrams for
          each doc
        - Gets the frequency of each bigram and trigram
          generated.
    """

    # Get the frequency of each word for each document
    docs_pdf['word_frequencies'] = docs_pdf.doc_words.apply(
        nltk.FreqDist
    )

    get_bigrams = lambda doc_words: [
        bigram for bigram in ngrams(doc_words, 2)
    ]

    get_trigrams = lambda doc_words: [
        trigram for trigram in ngrams(doc_words, 3)
    ]

    # Generate all bigrams and trigrams
    docs_pdf['bigrams'] = docs_pdf.doc_words.apply(get_bigrams)
    docs_pdf['trigrams'] = docs_pdf.doc_words.apply(get_trigrams)
    
    # Get the bigram and trigram frequency
    docs_pdf['bigrams_freq'] = docs_pdf.bigrams.apply(Counter)
    docs_pdf['trigrams_freq'] = docs_pdf.trigrams.apply(Counter)


def get_summary_for_iterable(
    word: str,
    docs_pdf: DataFrame,
    iterable_col: str
) -> Dict:
    """
    Takes word string, document dataframe and
    a string that represents an iterable column
    in the dataframe.
    Examples of these iterable columns include
    the sentences column and the bigram and trigram
    columns.
    Gets the occurrences of the word for each object in
    the iterable column for each document. Both count
    and returns each of these occurrences and also returns
    the total number of occurrences across all the docs.
    For example for the word 'let' in the 'sentences' column.
    We'll return the number of times 'let' comes up in a sentence
    for each doc, each of these sentences, and the total number
    of times 'let' comes up in a sentence across every doc.
    """

    freq_dict = {
        doc:0 for doc in docs_pdf.doc_name
    }

    # Gather all sentences the word occurs in for each doc
    occurances = [
        (row.doc_name, obj) for index, row in docs_pdf.iterrows() for obj in row[iterable_col]
        if word in obj
    ]

    # Get the number of sentence occurances for the word in each doc
    for obj in occurances:
        freq_dict[obj[0]] += 1

    # total number of sentences occurances for the word
    total_frequency = sum(freq_dict.values())
    
    return {
        "num_occurances_in_{}_by_doc".format(iterable_col): freq_dict,
        "total_{}_occurances".format(iterable_col[:-1]): total_frequency,
        iterable_col + "_occurances": occurances
    }


def get_full_word_summary(
    docs_pdf: DataFrame,
    word: str
) -> Dict:
    """
    Takes the document dataframe
    and a word as args. Gets the frequency/
    the number of occurrences of this word in each doc.
    Then generates a get_summary_for_iterable() on the
    sentences, bigram, and trigram columns, producing
    the word summary explained above.
    """

    word_freq_dict = {
        row['doc_name']:row['word_frequencies'][word]
        for index, row in docs_pdf.iterrows() if word in row['word_frequencies']
    }
    total_occurances = sum(word_freq_dict.values())

    sentence_summary = get_summary_for_iterable(word, docs_pdf, 'sentences')
    bigram_summary = get_summary_for_iterable(word, docs_pdf, 'bigrams')
    trigram_summary = get_summary_for_iterable(word, docs_pdf, 'trigrams')

    return {
        "word": word,
        "num_occurances_by_doc": word_freq_dict,
        "total_occurances": total_occurances,
        **{
            key:value
            for summary in [sentence_summary, bigram_summary, trigram_summary]
            for key, value in summary.items()
        }
    }


def get_tf_idf_matrix(
    docs_pdf: DataFrame
) -> DataFrame:
    """
    Takes the doc_pdf as an argument
    returns the TF-IDF matrix for
    the corpus of documents in the
    document dataframe. fit_transform() takes
    a document string as an argument. Seeing
    as we want to do our own preprocessing we
    can do this beforehand then join the words
    with the space delimiter to mimic a docstring.
    The TF-IDF matrix gives us an indication of the
    importance of a given word to a document in a corpus.
    It helps us find the words to pay attention to.
    """

    vectorizer = TfidfVectorizer()
    response = vectorizer.fit_transform(
        [
            ' '.join(words) for words in docs_pdf.doc_words
        ]
    )
    tf_idf = pd.DataFrame(response.toarray(), columns=vectorizer.get_feature_names_out())
    return tf_idf


def is_word_of_interest(
    word: str,
    pos_type: str = "noun"
) -> bool:
    """
    Takes word and part of speech as
    agrs. Returns true if the word belongs
    to the required POS.
    """
    if pos_type == 'noun':
        post_list = ["NN", "NNP", "NNPS", "NST", "NNS"]
    elif pos_type in ["adjective", "verb"]:
        post_list = [
            "JJ", "JJR", "JJS", "RB", "RBR", "RBS",
            "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"
        ]

    return (
        nltk.pos_tag([word])[0][1] in post_list
    )


def get_summaries_for_unique_words(
    docs_pdf: DataFrame
) -> DataFrame:
    """
    Takes the document dataframe as
    an arg. Uses the get_full_word_summary()
    function explained above and gets a word
    summary for every unique word in the corpus
    """

    unique_words = set([word for words in docs_pdf.doc_words for word in words])
    word_summaries = []
    for word in unique_words:
        word_summaries.append(
            get_full_word_summary(
                docs_pdf=docs_pdf,
                word=word
            )
        )
    return pd.DataFrame(word_summaries)


def get_docs_sorted_tf_idf_lookup_dicts(
    tf_idf: DataFrame,
    docs_pdf: DataFrame
) -> Dict:
    """
    Takes the TF-IDF matrix and documents
    dataframe as args. Returns a dictionary
    of document name to a dictionary of
    word: TF-IDF key values pairs sorted by
    TF-IDF value. This dictionary is used as
    a quick lookup or search to find the most
    important words for a document
    """

    tf_idf_lookup_dict = {}
    get_doc_name = lambda name: dict(docs_pdf.doc_name)[name]

    for index, row in tf_idf.iterrows():
        tf_idf_lookup_dict[get_doc_name(row.name)] = {
            k: v for k, v in sorted(dict(row).items(), key=lambda item: item[1], reverse=True)
        }
    return tf_idf_lookup_dict


def generate_top30_interesting_words(
    doc_name: str,
    tf_idf_lookup_dict: DataFrame,
    pos_type: str
) -> None:
    """
    Takes a document name, the TF-IDF
    lookup dict generated above and the
    part of speech tag we're interested in.
    Finds the top 30 words with the give POS.
    Th ranking here is the TF-IDF value.
    Writes a dataframe to disk that contains
    a word summary for each of these top 30 words
    and a TF-IDF column for each of the documents
    in the corpus.
    """

    interesting_words = []
    for word in tf_idf_lookup_dict[doc_name]:
        if len(interesting_words) == 30:
            break
        else:
            if is_word_of_interest(word, pos_type=pos_type):
                interesting_words.append(word)

    pd.DataFrame(
        [
            {
                "tf_idf_value": tf_idf_lookup_dict[doc_name][word],
             **get_full_word_summary(docs_pdf, word=word)
            } for word in interesting_words
        ]
    ).to_csv(
        "files/outputs/document_specific/{}_top30_interesting_{}s.csv".format(
            doc_name,
            pos_type
        )
    )


if __name__ == "__main__":

    # Download required packages
    setup_script()
    file_paths = glob.glob("./files/inputs/*.txt")
    docs_pdf = extract_documents(file_paths=file_paths)

    # Inital Cleaning and Preprocessing of documents
    split_docs_to_sentences(docs_pdf=docs_pdf)
    tokenize_document_words(docs_pdf=docs_pdf)
    clean_and_preprocess_docs(docs_pdf=docs_pdf)

    # Write inital document summaries to a csv
    docs_pdf.to_csv("files/outputs/overall_document_summary.csv")

    # Write a summary of all unique words in the corpus
    generate_word_summary_data(docs_pdf=docs_pdf)
    word_summary_pdf = get_summaries_for_unique_words(docs_pdf=docs_pdf)
    word_summary_pdf.to_csv("files/outputs/all_words_summary.csv")

    # Generate a TF-IDF matrix for the corpus and write to csv
    tf_idf = get_tf_idf_matrix(docs_pdf=docs_pdf)
    tf_idf.to_csv("files/outputs/tf_idf_matrix.csv")

    # Create a lookup dict for the word to TF-IDF values
    tf_idf_lookup_dict = get_docs_sorted_tf_idf_lookup_dicts(
        tf_idf=tf_idf,
        docs_pdf=docs_pdf
    )
    # Generate word summaries for each of the top 30
    # Nouns and [adjectives,adverbs,verbs] and write to csv
    for doc_name in tf_idf_lookup_dict.keys():
        generate_top30_interesting_words(
            doc_name=doc_name,
            tf_idf_lookup_dict=tf_idf_lookup_dict,
            pos_type="noun"
        )
    for doc_name in tf_idf_lookup_dict.keys():
        generate_top30_interesting_words(
            doc_name=doc_name,
            tf_idf_lookup_dict=tf_idf_lookup_dict,
            pos_type="adjective"
        )