"""
Processor for performing named entity tagging.
"""
import logging

from classla.models.common import doc
from classla.models.common.utils import unsort
from classla.models.ner.data import DataLoader
from classla.models.ner.trainer import Trainer
from classla.models.ner.inference import predict_batches
from classla.pipeline._constants import *
from classla.pipeline.processor import UDProcessor, register_processor

logger = logging.getLogger('classla')

@register_processor(name=NER)
class NERProcessor(UDProcessor):

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([NER])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([TOKENIZE])

    def _set_up_model(self, config, use_gpu):
        # set up trainer
        args = {'charlm_forward_file': config['forward_charlm_path'], 'charlm_backward_file': config['backward_charlm_path']}
        self._trainer = Trainer(args=args, model_file=config['model_path'], use_cuda=use_gpu)
        self._trainer.decoder = config.get('decoder', 'numpy')
        if self._trainer.decoder not in ('numpy', 'numba'):
            raise ValueError('NER decoder must be numpy or numba')

    def process(self, document):
        # set up a eval-only data loader and skip tag preprocessing
        batch = DataLoader(
            document, self.config['batch_size'], self.config, vocab=self.vocab, evaluation=True, preprocess_tags=False,
            inference_only=True, sort_during_eval=self.config.get('sort_during_eval', False))
        preds = []
        for prediction in predict_batches(self.trainer, batch, self.config.get('pipeline_depth', 1)):
            preds += prediction
        preds = unsort(preds, batch.data_orig_idx)
        batch.doc.set([doc.NER], [y for x in preds for y in x], to_token=True)
        # collect entities into document attribute
        total = len(batch.doc.build_ents())
        logger.debug(f'{total} entities found in document.')
        return batch.doc
