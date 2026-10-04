# RL inputs

R1Searcher_filtered.json contains the 4,030 RL training questions, sampled from the R1-Searcher training data. HotpotQA_val.json and 2wiki_val.json are the checkpoint-selection sets, 128 questions each. They are disjoint from the training questions and from the evaluation questions in benchmarks/.

Each record contains a question and an answer (a string or a list of answer aliases):

    {"question": "Who wrote The Hobbit?", "answer": ["J. R. R. Tolkien", "Tolkien"]}
