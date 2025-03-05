import pandas as pd

filepath = "NRC-emotion-lexicon.txt"
emolex_df = pd.read_csv(filepath,  names=["word", "emotion", "association"], skiprows=45, sep='\t', keep_default_na=False)
emolex_words = emolex_df.pivot(index='word', columns='emotion', values='association').reset_index()
print(emolex_words[emolex_words.anger == 1].word)