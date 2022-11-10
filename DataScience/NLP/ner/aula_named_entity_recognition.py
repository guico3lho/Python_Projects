import pandas as pd

def preprocessDataFrame(df):
    dic = {}
    dic['tokens'] = []

    for text in df['tokens']:
        tokens = []
        for x in text:
            tokens.append(x.decode('utf-8'))
        l = " ".join(tokens)
        dic['tokens'].append(l.split())

    res_df = pd.DataFrame.from_dict(dic)
    res_df['ner'] = df['ner']
    return res_df

def label2int():
    iob_labels = ["B", "I"]
    ner_labels = ["PER", "ORG", "LOC", "MISC"]
    all_labels = [(iob_label, ner_label) for ner_label in ner_labels for iob_label in iob_labels]
    all_labels = ["-".join([a, b]) for a, b in all_labels]
    dic = dict(zip(range(1, len(all_labels) + 1), all_labels))
    dic[0] = 'O'
    return dic