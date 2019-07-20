# Document Retrieval System

Vector-space calculations to allow for retrieval under different term weighting schemes

## What's Included

### Data Files

The materials provided include a file documents.txt, which contains a collection of documents that record publications in the CACM (Communications of the Association for Computing Machinery). Each document is a short record of a CACM paper, including its title, author(s), and abstract — although one or other of these (especially abstract) may be absent for a given document. The file queries.txt contains a set of IR queries for use against this collection. (These are ‘old-style’ queries, where users might write an entire paragraph describing their interest.) The file cacm gold std.txt is a ‘gold standard’ identifying the documents that have been judged relevant to each query. These three files together constitute a standard test set that has been used for evaluating IR systems (although it is now somewhat dated, not least by being very small by modern standards).

### Code Files

The materials provided include the code file ir engine.py, which is the ‘outer shell’ of a retrieval engine, that loads an index and preprocessed query set, and then ‘batch processes’ the queries, i.e. uses the index to compute the 10 best-ranking documents for each COM3110 / COM4115 / COM6115 Page 1 Assignment: 2018/19 query, which it prints to a results file. Run this program with its help option (-h) for information on its command line options. These include flags for whether stoplisting and/or stemming are applied during preprocessing (which are used to determine which of the index and query files to load). Another option allows the user to set the name of the file to which results are written. A final option allows the user to select the term weighting scheme used during retrieval, with a choice of binary, tf (term frequency) and tfidf modes.

The Python script eval ir.py calculates system performance scores, by comparing the collection gold standard (cacm gold std.txt) to a system results file (which lists the ids of the documents the system returns for each query). Execute the script with its help option (-h) for instructions on use.

The program ir engine.py can be executed to generate a results file, but you will find that it scores zero for retrieval performance. The program does implement various aspects of required functionality, i.e. it processes the command line, loads the selected index file into a suitable data structure (a two-level dictionary), loads the preprocessed queries, runs a batch process over the queries, and prints out the results to file. However, it does not include a sensible implementation of the functionality for computing what are the most relevant documents for a given query, based on the index. This functionality is to be provided by the class Retrieve which ir engine.py imports from the file my retriever.py, but the current definition provided in that file is just a ‘stub’ which returns the same result for every query (which is just a list of the numbers 1 to 10, as if these were the ids of the documents selected as relevant).
