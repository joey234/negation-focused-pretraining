import torch
from model import RobertaForFourWayTokenTypeClassification, RobertaForTwoWayTokenTypeClassification
from transformers import RobertaTokenizer
from transformers import AutoModelForTokenClassification, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained('roberta-base')

model = AutoModelForTokenClassification.from_pretrained('roberta-base')

# sents = 'No serious complications such as hypertension, diabetes, coronary heart disease and psychiatric history.'
sents = 'Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO'
inputs = tokenizer(sents, return_tensors = "pt")
tokens = inputs.tokens()

# output = model(inputs['input_ids'])
# print(output)
outputs = model(**inputs).logits
predictions = torch.argmax(outputs, dim=2)

for token, prediction in zip(tokens, predictions[0].numpy()):
	print((token, model.config.id2label[prediction]))