import torch, os
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from tqdm import tqdm

spl_tokens = {'cont_start': '<CONT_START>',
                  'attr_start': '<ATTR_START>',
                  'data_start': '<DATA_START>',
                  'end': '<END>'}
special_tokens = [spl_tokens[i] for i in spl_tokens] # Set the special tokens

tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt', special_tokens=special_tokens)
device = torch.device('cuda')
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt', num_special_tokens=len(special_tokens))
path = os.path.join(os.getcwd(),'../runs/yelp_3_epoch/pytorch_model_1.bin')
model_state_dict = torch.load(path)
model.load_state_dict(model_state_dict)
model.to(device)
model.eval()
end_id = tokenizer.special_tokens['<END>']


def prediction(ref_text):
    predicted_index = None
    decoded_sentence = []
    tokens = tokenizer.tokenize(ref_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
    while True:
        torch_tokens = torch.tensor([indexed_tokens]).to(device)
        with torch.no_grad():
            prediction = model(torch_tokens)
        predicted_index = torch.argmax(prediction[0, -1, :]).item()
        if predicted_index == end_id:
            break
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        decoded_sentence.append(predicted_index)
        indexed_tokens.append(predicted_index)

    return tokenizer.decode(decoded_sentence)

DATA = '/home/ubuntu/data/amazon'
POS_TEST_FILE_PATH = os.path.join(DATA, 'processed.reference.inputs.from.1')
NEG_TEST_FILE_PATH = os.path.join(DATA,'processed.reference.inputs.from.0')
POS_TEST_FILE_OUT_PATH = os.path.join('../runs/amazon_1_plus_3_epoch/sentiment.reference.out.from.1')
NEG_TEST_FILE_OUT_PATH = os.path.join('../runs/amazon_1_plus_3_epoch/sentiment.reference.out.from.0')

POS_REF_FILE_PATH = os.path.join(DATA, 'processed.sentiment.test.inputs.1')
NEG_REF_FILE_PATH = os.path.join(DATA,'processed.sentiment.test.inputs.0')
POS_REF_FILE_OUT_PATH = os.path.join('../runs/amazon_1_plus_3_epoch/sentiment.test.out.1')
NEG_REF_FILE_OUT_PATH = os.path.join('../runs/amazon_1_plus_3_epoch/sentiment.test.out.0')

# DATA = '/home/ubuntu/data/yelp'
# POS_TEST_FILE_PATH = os.path.join(DATA, 'processed.reference.inputs.from.1')
# NEG_TEST_FILE_PATH = os.path.join(DATA,'processed.reference.inputs.from.0')
# POS_TEST_FILE_OUT_PATH = os.path.join('../runs/yelp_3_epoch/sentiment.reference.out.from.1')
# NEG_TEST_FILE_OUT_PATH = os.path.join('../runs/yelp_3_epoch/sentiment.reference.out.from.0')
#
# POS_REF_FILE_PATH = os.path.join(DATA, 'processed.sentiment.test.inputs.1')
# NEG_REF_FILE_PATH = os.path.join(DATA,'processed.sentiment.test.inputs.0')
# POS_REF_FILE_OUT_PATH = os.path.join('../runs/yelp_3_epoch/sentiment.test.out.1')
# NEG_REF_FILE_OUT_PATH = os.path.join('../runs/yelp_3_epoch/sentiment.test.out.0')

with open(POS_TEST_FILE_PATH, 'r') as f:
    sents_pos_test = f.readlines()
with open(NEG_TEST_FILE_PATH, 'r') as f:
    sents_neg_test = f.readlines()
with open(POS_REF_FILE_PATH, 'r') as f:
    sents_pos_ref = f.readlines()
with open(NEG_REF_FILE_PATH, 'r') as f:
    sents_neg_ref = f.readlines()
sents_pos_test = list(map(lambda x: x.strip(), sents_pos_test))
sents_neg_test = list(map(lambda x: x.strip(), sents_neg_test))
sents_pos_ref = list(map(lambda x: x.strip(), sents_pos_ref))
sents_neg_ref= list(map(lambda x: x.strip(), sents_neg_ref))

print('Predicting for pos test sents..')
sents_pos_out_test = [prediction(i) for i in sents_pos_test]
print('Predicting for neg test sents..')
sents_neg_out_test = [prediction(i) for i in sents_neg_test]
print('Predicting for pos ref sents..')
sents_pos_out_ref = [prediction(i) for i in sents_pos_ref]
print('Predicting for neg ref sents..')
sents_neg_out_ref = [prediction(i) for i in sents_neg_ref]

print('Writing to out file..')
with open(POS_TEST_FILE_OUT_PATH, 'w') as f:
    f.write('\n'.join(sents_pos_out_test))
    f.close()
with open(NEG_TEST_FILE_OUT_PATH, 'w') as f:
    f.write('\n'.join(sents_neg_out_test))
    f.close()
with open(POS_REF_FILE_OUT_PATH, 'w') as f:
    f.write('\n'.join(sents_pos_out_ref))
    f.close()
with open(NEG_REF_FILE_OUT_PATH, 'w') as f:
    f.write('\n'.join(sents_neg_out_ref))
    f.close()

