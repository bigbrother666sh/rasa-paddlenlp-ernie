from paddlenlp.transformers import (
    ErnieTokenizer,
    ErnieModel,
    BertModel,
    BertTokenizer,
    XLNetModel,
    XLNetTokenizer,
    RobertaModel,
    RobertaTokenizer,
)

# these seems to be useful still, keeping
from rasa.nlu.utils.hugging_face.transformers_pre_post_processors import (
    bert_tokens_pre_processor,
    gpt_tokens_pre_processor,
    xlnet_tokens_pre_processor,
    roberta_tokens_pre_processor,
    bert_embeddings_post_processor,
    gpt_embeddings_post_processor,
    xlnet_embeddings_post_processor,
    roberta_embeddings_post_processor,
    bert_tokens_cleaner,
    openaigpt_tokens_cleaner,
    gpt2_tokens_cleaner,
    xlnet_tokens_cleaner,
)

model_special_tokens_pre_processors = {
    "bert": bert_tokens_pre_processor,
    "ernie": bert_tokens_pre_processor,
    "gpt": gpt_tokens_pre_processor,
    "gpt2": gpt_tokens_pre_processor,
    "xlnet": xlnet_tokens_pre_processor,
    # "xlm": xlm_tokens_pre_processor,
    "distilbert": bert_tokens_pre_processor,
    "roberta": roberta_tokens_pre_processor,
}

model_tokens_cleaners = {
    "bert": bert_tokens_cleaner,
    "ernie": bert_tokens_cleaner,
    "gpt": openaigpt_tokens_cleaner,
    "gpt2": gpt2_tokens_cleaner,
    "xlnet": xlnet_tokens_cleaner,
    # "xlm": xlm_tokens_pre_processor,
    "distilbert": bert_tokens_cleaner,  # uses the same as BERT
    "roberta": gpt2_tokens_cleaner,  # Uses the same as GPT2
}

model_embeddings_post_processors = {
    "bert": bert_embeddings_post_processor,
    "ernie": bert_embeddings_post_processor,
    "gpt": gpt_embeddings_post_processor,
    "gpt2": gpt_embeddings_post_processor,
    "xlnet": xlnet_embeddings_post_processor,
    # "xlm": xlm_embeddings_post_processor,
    "distilbert": bert_embeddings_post_processor,
    "roberta": roberta_embeddings_post_processor,
}

model_class_dict = {
    "bert": BertModel,
    "ernie": ErnieModel,
    "xlnet": XLNetModel,
    "roberta": RobertaModel,
}

model_tokenizer_dict = {
    "bert": BertTokenizer,
    "ernie": ErnieTokenizer,
    "xlnet": XLNetTokenizer,
    "roberta": RobertaTokenizer,
}

model_weights_defaults = {
    "bert": "bert-wwm-ext-chinese",
    "ernie": "ernie-3.0-base-zh",
    "xlnet": "chinese-xlnet-base",
    "roberta": "roberta-wwm-ext",
}
