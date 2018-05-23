import tensorflow as tf

from sequence_encoder import SequenceEncoder

class SequenceEmbedder:

    def __init__(self, vocab_size, embedding_size, hidden_size, batch_size, dtype):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.dtype = dtype

        self._build_model()

    def _build_model(self):
        self.embeddings = tf.get_variable(
            'word_embeddings',
            [self.vocab_size, self.embedding_size])

    def embed(self, inputs, input_lengths):
        inputs = tf.nn.embedding_lookup(self.embeddings, inputs)
        tf.summary.histogram(
            'characters_length',
            input_lengths)

        encoder = SequenceEncoder(self.hidden_size, self.dtype, self.batch_size)
        hidden_state = encoder.encode(inputs)

        return hidden_state
