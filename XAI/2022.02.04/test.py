from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-squadv2")
model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-squadv2")


def get_answer(question, context):
    input_text = "question: %s  context: %s" % (question, context)
    features = tokenizer([input_text], return_tensors='pt')

    output = model.generate(input_ids=features['input_ids'],
                            attention_mask=features['attention_mask'])

    return tokenizer.decode(output[0])


context = "In Norse mythology, Valhalla is a majestic, enormous hall located in Asgard, ruled over by the god Odin."
question = "What is Valhalla ?"

answer = get_answer(question, context)
print(answer)


context = "The economy of Victoria is highly diversified: service sectors including financial and property services, health, education, wholesale, retail, hospitality and manufacturing constitute the majority of employment. Victoria's total gross state product (GSP) is ranked second in Australia, although Victoria is ranked fourth"
question = "What kind of economy does Victoria have?"

answer = get_answer(question, context)
print(answer)
print("finished")
# output: 'HF-Transformers and Google'