import pandas as pd
from bert_serving.client import BertClient

def vectorize(sentence_series, ip='localhost', port=5555):
    bc = BertClient(show_server_config=True , ignore_all_checks=True, port=port, ip=ip)
    return list(bc.encode(sentence_series.tolist()))
    