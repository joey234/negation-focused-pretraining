from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./roberta-maskneg-bioscope",
    tokenizer="roberta-base"
)
# print(fill_mask("the patient was <mask> tested for diabetes."))
print(fill_mask("<mask> serious complications such as hypertension, diabetes, coronary heart disease and psychiatric history."))
# sents = 'No serious complications such as hypertension, diabetes, coronary heart disease and psychiatric history.'
