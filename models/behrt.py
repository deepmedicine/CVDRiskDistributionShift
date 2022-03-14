from models.parts.blocks import BertPooler, BertEncoder, BertLayerNorm
from models.parts.embeddings import Embedding
import torch.nn as nn
import pytorch_lightning as pl
import torch
import pytorch_pretrained_bert as Bert
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.utils import load_obj
from torch.optim import *
from optim.tri_stage_lr_scheduler import TriStageLRScheduler
from torchmetrics.functional import average_precision, auroc
import os


class Behrt(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params

        vocab_size = len(load_obj(self.params['token_dict_path'])['token2idx'].keys())
        age_size = len(load_obj(self.params['age_dict_path'])['token2idx'].keys())
        self.params.update({'vocab_size': vocab_size, 'age_vocab_size': age_size})

        self.save_hyperparameters()

        self.feature_extractor = BEHRT(params)
        self.classifier_head = ClassifierHead(params)

        self.sig = nn.Sigmoid()

        self.apply(self.init_bert_weights)

        if params['checkpoint_feature'] is not None:
            pretrained_dict = torch.load(params['checkpoint_feature'],
                                         map_location=lambda storage, loc: storage)['state_dict']
            model_dict = self.state_dict()

            print('incompatible keys: ', {k for k,v in pretrained_dict.items() if k not in model_dict})

            if params['embedding_only']:
                embedding_list = ['feature_extractor.embedding.word_embeddings.weight',
                                  'feature_extractor.embedding.segment_embeddings.weight',
                                  'feature_extractor.embedding.age_embeddings.weight',
                                  'feature_extractor.embedding.posi_embeddings.weight',
                                  'feature_extractor.embedding.LayerNorm.weight',
                                  'feature_extractor.embedding.LayerNorm.bias']
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in embedding_list}
            else:
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.load_state_dict(model_dict)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.params['initializer_range'])
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, record, age, seg, position, att_mask):
        representation = self.feature_extractor(record, age, seg, position, att_mask)
        y = self.classifier_head(representation)
        return representation, y

    def shared_step(self, batch, batch_idx):
        record,  age, seg, position, att_mask, label = \
            batch['code'], batch['age'], batch['seg'], batch['position'], batch['att_mask'], batch['label']

        _, y = self.forward(record, age, seg, position, att_mask)

        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(y.view(-1, 1), label.view(-1, 1))

        return loss, y, label

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.shared_step(batch, batch_idx)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self.shared_step(batch, batch_idx)
        # log results
        self.log("val_loss", loss)

    def configure_optimizers(self):
        if self.params['optimiser'] == 'Adam':
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

            optimizer_grouped_parameters = [
                {'params': [p for n, p in list(self.named_parameters()) if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.params['optimiser_params']['weight_decay']},
                {'params': [p for n, p in list(self.named_parameters()) if any(nd in n for nd in no_decay)],
                 'weight_decay': 0}
            ]

            optimizer = Bert.optimization.BertAdam(optimizer_grouped_parameters,
                                                   lr=self.params['optimiser_params']['lr'],
                                                   warmup=self.params['optimiser_params']['warmup_proportion'])
        elif self.params['optimiser'] == 'SGD':
            optimizer = SGD(self.parameters(), lr=self.params['optimiser_params']['lr'], momentum=self.params['optimiser_params']['momentum'])
        else:
            raise ValueError('the optimiser is not implimented')

        if self.params['lr_strategy'] == 'fixed':
            return optimizer
        elif self.params['lr_strategy'] == 'stri_stage':
            scheduler = TriStageLRScheduler(
                optimizer,
                **self.params['scheduler']
            )
            return [optimizer], [scheduler]

    def reset_buffer(self):
        self.pred_list = []
        self.target_list = []
        self.patid_list = []
        self.logits_list = []

    def on_test_epoch_start(self):
        self.reset_buffer()

    def test_step(self, batch, batch_idx):
        loss, y, label = self.shared_step(batch, batch_idx)
        patid = batch['patid']

        self.logits_list.append(y.cpu())
        self.pred_list.append(self.sig(y).cpu())
        self.target_list.append(label.cpu())
        self.patid_list.append(patid.cpu())

    def test_epoch_end(self, outs):

        label = torch.cat(self.target_list, dim=0).view(-1)
        pred = torch.cat(self.pred_list, dim=0).view(-1)
        patid = torch.cat(self.patid_list, dim=0).view(-1)
        logits = torch.cat(self.logits_list, dim=0).view(-1)

        auprc_score = average_precision(pred, target=label.type(torch.LongTensor))
        auroc_score = auroc(pred, label.type(torch.LongTensor))

        result = {
            'patid': patid,
            'label': label,
            'prediction': pred,
            'logits': logits
        }

        print('epoch : {} AUROC: {} AUPRC: {}'.format(self.current_epoch, auroc_score, auprc_score))
        torch.save(result, os.path.join(self.params['save_path'], 'result.pt'))


class Extractor(nn.Module):
    def __init__(self, params):
        super(Extractor, self).__init__()
        self.encoder = BertEncoder(params, params['extractor_num_layer'])
        self.pooler = BertPooler(params)

    def forward(self, hidden_state, mask, encounter=True):
        mask = mask.to(dtype=next(self.parameters()).dtype)

        attention_mast = mask.unsqueeze(2).unsqueeze(3)

        attention_mast = (1.0 - attention_mast) * -10000.0

        encoded_layer = self.encoder(hidden_state, attention_mast, encounter)
        encoded_layer = self.pooler(encoded_layer, encounter)

        encode_visit = encoded_layer
        return encode_visit  # [batch * seg_len * Dim]


class Aggregator(nn.Module):
    def __init__(self, params):
        super(Aggregator, self).__init__()
        self.encoder = BertEncoder(params, params['aggregator_num_layer'])

    def forward(self, hidden_state, mask, encounter=True):
        mask = mask.to(dtype=next(self.parameters()).dtype)
        attention_mast = mask.unsqueeze(1).unsqueeze(2)
        attention_mast = (1.0 - attention_mast) * -10000.0
        encoded_layer = self.encoder(hidden_state, attention_mast, encounter)

        return encoded_layer  # batch seq_len dim


class BEHRT(nn.Module):
    def __init__(self, params):
        super(BEHRT, self).__init__()
        self.params = params
        self.embedding = Embedding(params)
        # self.extractor = Extractor(params)
        self.aggregator = Aggregator(params)

    def forward(self, record, age, seg, position, att_mask):
        output = self.embedding(record, age, seg, position)
        # output = self.extractor(output, att_mask, encounter=True)
        h = self.aggregator(output, att_mask, encounter=False)
        return h


class ClassifierHead(nn.Module):
    def __init__(self, params):
        super(ClassifierHead, self).__init__()
        self.pooler = BertPooler(params)
        self.classifier = nn.Linear(in_features=params['hidden_size'], out_features=1)

    def forward(self, x):
        x = self.pooler(x, encounter=False)
        output = self.classifier(x)
        return output