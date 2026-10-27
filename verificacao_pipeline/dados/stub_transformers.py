"""Substituto mínimo de `transformers` para correr create_embeddings.py sem descarregar o GraphCodeBERT (~500 MB).

É copiado para a pasta de trabalho com o nome `transformers.py`, para que o `import transformers`
do script original o encontre primeiro. Devolve vetores aleatórios de 768 dimensões: os testes
verificam o alinhamento, as etiquetas e o mapeamento, que não dependem do valor dos vetores.
"""
import torch


class _Inputs(dict):
    def to(self, device):
        return self


class RobertaTokenizer:
    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def __call__(self, texts, **kwargs):
        n, comprimento = len(texts), 4
        return _Inputs(input_ids=torch.zeros(n, comprimento, dtype=torch.long),
                       attention_mask=torch.ones(n, comprimento, dtype=torch.long))


class _Saida:
    def __init__(self, h):
        self.last_hidden_state = h


class RobertaModel:
    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        return _Saida(torch.rand(input_ids.shape[0], input_ids.shape[1], 768))
