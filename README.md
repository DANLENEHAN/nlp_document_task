Create python environment
- conda create -n eigen python=3.9

Activate environment
- conda activate eigen

Install Requirements
- pip install -r requirements.txt

Run proccess_docs file
- python process_docs.py


Task Summary:

My steps
-   I extract the docs from the files and create a pandas dataframe
    with each of the document names and raw documents as columns to start.
    This dataframe is used and built upon throughout.

-   Split each of the documents into sentences for later use.

-   Tokenize each of the documents so I can easily preprocess the
    docs. My cleaning and preprocessing consist of decapitating
    all the words and removing any stop words as I don't think they'll
    add any value to the analysis.

-   I write this initial work to a CSV file named 'files/outputs/overall_document_summary.csv'

-   I wrote some lemming and stemming functions but didn't end up using them as they produced
    some bad results and I felt I was losing some information doing this.

-   I created a word summary for every unique word in the corpus.
    The word summary for all the words can be found in the all_words_summary.csv.
    Essentially for each word I get the number of occurrences for each word in each
    document and the total occurrences in all the documents. I then get the number
    of sentences each word appears in for each document as well as an overall count
    and I keep a list for these sentences. I generate all the bigrams and trigrams for
    each document in a preparation step. I then, like with the sentences get all the bigrams
    and trigrams each word appears in, keep a count per doc as well as an overall count.
    I add all this data to a pandas dataframe and run the process for each word.

-   I'll then generate a TF-IDF matrix. This will help me find the most important words for
    each of the docs in the corpus. I create the matrix write it to a CSV and then also create
    a lookup for this matrix. A lookup is a dictionary of document name to a dictionary of
    word: TF-IDF key values pairs sorted by TF-IDF value.

    This sorted dictionary gives me all the words for each document sorted by importance. There's
    one more step I did to find the most important words. In this TF-IDF matrix the word 'let' ranked
    quite high for document 1 but I felt this wasn't telling me much. To help with this I decided I'd get
    the 30 most important Nouns and (Adjectives and Verbs). I created a function is_word_of_interest to
    return words that had the POS I was looking for.
    I gathered the top 30 most interesting nouns and adjectives for each document and wrote them all to
    files in the files/output/document_specific folder.
    These are the most important words for each of the documents and tell us the most information.


    If time permitted I would
        - Categorise each of the documents (give them a theme e.g News, Business, War, etc)
        - I'd also predict the sentiment of each of the documents.
        - I'd like to also dive more into the top 30 (or more) nouns. I'd loop through all
        the sentences, bigrams, and trigrams I saved for each of these nouns and try to predict
        the sentiment associated with the nouns.
        Being able to predict the sentiment for 'important nouns could be useful in the case
        of getting a view on sentiment towards companies for example which could help with stock
        predictions.