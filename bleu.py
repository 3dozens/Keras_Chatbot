from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

num_samples = 10000 # None = use all samples

with open("generated_from_train_data") as f:
    generated = f.readlines()

with open("./data_set/cornell movie-dialogs corpus/train.dec") as f:
    references = f.readlines()[:num_samples]

with open("./data_set/cornell movie-dialogs corpus/train.enc") as f:
    enc = f.readlines()[:num_samples]

# for gen, ref, enc in zip(generated, references, enc):
#     print("Q =", enc)
#     print("A =", ref)
#     print("Bot's return = ", gen)
#     print("---------------")

generated  = [ word_tokenize(gen)  for gen in generated]
references = [[word_tokenize(ref)] for ref in references] # one reference per one generated sentence

print(corpus_bleu(references, generated, smoothing_function=SmoothingFunction().method4))
