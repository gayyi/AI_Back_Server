import sentencepiece as spm

spm.SentencePieceTrainer.train(input = 'AI_Back_Server/Project2/Sources/Sports_shoes_knowledge_base.txt',model_prefix = 'tokenizer', vocab_size = 310)

###加载训练好的分词器
sp = spm.SentencePieceProcessor(model_file = 'tokenizer.model')

#分词提示
example = "this is a test sentence"
tokens = sp.encode(example, out_type =str)
token_ids = sp.encode(example, out_type = int)
print(tokens)
print(token_ids)
