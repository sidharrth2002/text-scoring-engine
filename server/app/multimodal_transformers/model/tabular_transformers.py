import torch
from torch import nn
from torch.utils.data.dataset import T
from transformers import (
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    DistilBertForSequenceClassification,
    AlbertForSequenceClassification,
    XLNetForSequenceClassification,
    XLMForSequenceClassification,
    LongformerForSequenceClassification,
)
from transformers.models.bert.modeling_bert import BERT_INPUTS_DOCSTRING
from transformers.models.roberta.modeling_roberta import ROBERTA_INPUTS_DOCSTRING
from transformers.models.distilbert.modeling_distilbert import DISTILBERT_INPUTS_DOCSTRING
from transformers.models.albert.modeling_albert import ALBERT_INPUTS_DOCSTRING
from transformers.models.xlnet.modeling_xlnet import XLNET_INPUTS_DOCSTRING
from transformers.models.xlm.modeling_xlm import XLM_INPUTS_DOCSTRING
from transformers.models.longformer.modeling_longformer import LONGFORMER_INPUTS_DOCSTRING
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaConfig
from transformers.file_utils import add_start_docstrings

from .layers import KeyAttention, LambdaLayer

from .tabular_combiner import TabularFeatCombiner
from .tabular_config import TabularConfig
from .layer_utils import MLP, calc_mlp_dims, hf_loss_func

import pickle

class BertWithTabular(BertForSequenceClassification):
    """
    Bert Model transformer with a sequence classification/regression head as well as
    a TabularFeatCombiner module to combine categorical and numerical features
    with the Bert pooled output

    Parameters:
        hf_model_config (:class:`~transformers.BertConfig`):
            Model configuration class with all the parameters of the model.
            This object must also have a tabular_config member variable that is a
            :obj:`TabularConfig` instance specifying the configs for :obj:`TabularFeatCombiner`
    """

    def __init__(self, hf_model_config):
        super().__init__(hf_model_config)
        tabular_config = hf_model_config.tabular_config
        if type(tabular_config) is dict:  # when loading from saved model
            tabular_config = TabularConfig(**tabular_config)
        else:
            self.config.tabular_config = tabular_config.__dict__

        self.add_attention_module = tabular_config.add_attention_module
        self.save_attentions = tabular_config.save_attentions
        self.attentions_path = tabular_config.attentions_path

        tabular_config.text_feat_dim = hf_model_config.hidden_size
        tabular_config.hidden_dropout_prob = hf_model_config.hidden_dropout_prob
        self.tabular_combiner = TabularFeatCombiner(tabular_config)
        self.num_labels = tabular_config.num_labels
        combined_feat_dim = self.tabular_combiner.final_out_dim
        self.dropout = nn.Dropout(hf_model_config.hidden_dropout_prob)
        self.batch_size = tabular_config.batch_size

        if self.add_attention_module:
            self.keyword_MLP = MLP(300 * tabular_config.num_keywords * 2, tabular_config.keyword_MLP_out_dim, num_hidden_lyr=1, dropout_prob=0.1, hidden_channels=[600], bn=True)

        if tabular_config.use_simple_classifier:
            if self.add_attention_module:
                self.tabular_classifier = nn.Linear(combined_feat_dim,
                                                    tabular_config.num_labels)
            else:
                self.tabular_classifier = nn.Linear(combined_feat_dim, tabular_config.num_labels)
        else:
            if self.add_attention_module:
                dims = calc_mlp_dims(combined_feat_dim,
                                    division=tabular_config.mlp_division,
                                    output_dim=tabular_config.num_labels)
                self.tabular_classifier = MLP(combined_feat_dim,
                                            tabular_config.num_labels,
                                            num_hidden_lyr=len(dims),
                                            dropout_prob=tabular_config.mlp_dropout,
                                            hidden_channels=dims,
                                            bn=True)
            else:
                dims = calc_mlp_dims(combined_feat_dim,
                                    division=tabular_config.mlp_division,
                                    output_dim=tabular_config.num_labels)
                self.tabular_classifier = MLP(combined_feat_dim,
                                            tabular_config.num_labels,
                                            num_hidden_lyr=len(dims),
                                            dropout_prob=tabular_config.mlp_dropout,
                                            hidden_channels=dims,
                                            bn=True)

        # dangerous harcoding, fix later
        if self.add_attention_module:
            self.embedding_layer = nn.Embedding(num_embeddings=tabular_config.vocab_size, embedding_dim=300)

            self.att_layer = KeyAttention(
                name='attention',
                op='dp',
                seed=0,
                emb_dim=300,
                word_att_pool='mean',
                merge_ans_key='concat',
                beta=False,
                batch_size=self.batch_size,
                tabular_config=tabular_config
            )

    @add_start_docstrings(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        class_weights=None,
        output_attentions=None,
        output_hidden_states=None,
        cat_feats=None,
        numerical_feats=None,
        answer_tokens=None,
        keyword_tokens=None,
        answer_mask=None,
        keyword_mask=None,
        lemmatized_answer_tokens=None,
    ):
        r"""
        class_weights (:obj:`torch.FloatTensor` of shape :obj:`(tabular_config.num_labels,)`, `optional`, defaults to :obj:`None`):
            Class weights to be used for cross entropy loss function for classification task
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`tabular_config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`tabular_config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        cat_feats (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, tabular_config.cat_feat_dim)`, `optional`, defaults to :obj:`None`):
            Categorical features to be passed in to the TabularFeatCombiner
        numerical_feats (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, tabular_config.numerical_feat_dim)`, `optional`, defaults to :obj:`None`):
            Numerical features to be passed in to the TabularFeatCombiner
    Returns:
        :obj:`tuple` comprising various elements depending on configuration and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if tabular_config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, tabular_config.num_labels)`):
            Classification (or regression if tabular_config.num_labels==1) scores (before SoftMax).
        classifier_layer_outputs(:obj:`list` of :obj:`torch.FloatTensor`):
            The outputs of each layer of the final classification layers. The 0th index of this list is the
            combining module's output
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        if self.add_attention_module:
            ans_emb = self.embedding_layer(lemmatized_answer_tokens)
            ans_mask_emb = self.embedding_layer(answer_mask)
            keys_emb = self.embedding_layer(keyword_tokens)
            keys_mask_emb = self.embedding_layer(keyword_mask)

            att_rtn_keys = ('z', 'z_ans', 'z_key', 'beta_ans', 'beta_key')

            fea_att_list = list()
            attentions = {k: list() for k in att_rtn_keys}

            # was this causing CUDA error
            # first dimension is batch size
            for i in range(keyword_tokens.shape[1]):
                t_k = LambdaLayer(lambda x: x[:, i], name='key_%d' % i)(keys_emb)
                t_k_m = LambdaLayer(lambda x: x[:, i], name='ans_%d' % i)(keyword_mask)
                # t_k_m = LambdaLayer(lambda x: x[:, i], name='ans_%d' % i)(keyword_tokens)

                f, *att_rtn = self.att_layer([ans_emb, answer_mask, t_k, t_k_m])

                fea_att_list.append(f)

                for i_a_r, a_r in enumerate(att_rtn):
                    attentions[att_rtn_keys[i_a_r]].append(a_r)

            # do something with this- represents keyword attention
            # print('fea_att_list.shape', len(fea_att_list))
            # print('fea_att_list', fea_att_list)
            # print('Shapes of items in fea_att_list')
            # for item in fea_att_list:
            #     print(item.shape)
            fea_rubric = torch.cat(fea_att_list, dim=1)

            # print('fea_rubric shape', fea_rubric.shape)
            # print('fea_rubric', fea_rubric)

            layer_exp_dim = LambdaLayer(lambda x: torch.unsqueeze(x, 1), name='expand_dim')
            att_list = list()

            for k in attentions:
                att_k = attentions[k]
                att_exp = [layer_exp_dim(m) for m in att_k]
                if len(att_exp) == 1:
                    att_exp = att_exp[0]
                else:
                    att_exp = torch.cat(att_exp, dim=1)
                attentions[k] = att_exp
                att_list.append(att_exp)

            keyword_feats = self.keyword_MLP(fea_rubric.float())

            # straight away combine Text Features + Categorical Features + Numerical Features
            combined_feats = self.tabular_combiner(pooled_output,
                                               cat_feats,
                                               numerical_feats,
                                               keyword_feats=keyword_feats)

            loss, logits, classifier_layer_outputs = hf_loss_func(combined_feats,
                                                                self.tabular_classifier,
                                                                labels,
                                                                self.num_labels,
                                                                class_weights)
            if self.save_attentions:
                with open(self.attentions_path, 'wb') as handle:
                    pickle.dump(attentions, handle)

            return loss, logits, classifier_layer_outputs

        else:
            combined_feats = self.tabular_combiner(pooled_output,
                                               cat_feats,
                                               numerical_feats)

            loss, logits, classifier_layer_outputs = hf_loss_func(combined_feats,
                                                                self.tabular_classifier,
                                                                labels,
                                                                self.num_labels,
                                                                class_weights)
            return loss, logits, classifier_layer_outputs


class RobertaWithTabular(RobertaForSequenceClassification):
    """
    Roberta Model transformer with a sequence classification/regression head as well as
    a TabularFeatCombiner module to combine categorical and numerical features
    with the Roberta pooled output

    Parameters:
        hf_model_config (:class:`~transformers.RobertaConfig`):
            Model configuration class with all the parameters of the model.
            This object must also have a tabular_config member variable that is a
            :obj:`TabularConfig` instance specifying the configs for :obj:`TabularFeatCombiner`
    """
    def __init__(self, hf_model_config):
        super().__init__(hf_model_config)
        tabular_config = hf_model_config.tabular_config
        if type(tabular_config) is dict:  # when loading from saved model
            tabular_config = TabularConfig(**tabular_config)
        else:
            self.config.tabular_config = tabular_config.__dict__

        tabular_config.text_feat_dim = hf_model_config.hidden_size
        tabular_config.hidden_dropout_prob = hf_model_config.hidden_dropout_prob
        self.tabular_combiner = TabularFeatCombiner(tabular_config)
        self.num_labels = tabular_config.num_labels
        combined_feat_dim = self.tabular_combiner.final_out_dim
        self.dropout = nn.Dropout(hf_model_config.hidden_dropout_prob)
        if tabular_config.use_simple_classifier:
            self.tabular_classifier = nn.Linear(combined_feat_dim,
                                                tabular_config.num_labels)
        else:
            dims = calc_mlp_dims(combined_feat_dim,
                                 division=tabular_config.mlp_division,
                                 output_dim=tabular_config.num_labels)
            self.tabular_classifier = MLP(combined_feat_dim,
                                          tabular_config.num_labels,
                                          num_hidden_lyr=len(dims),
                                          dropout_prob=tabular_config.mlp_dropout,
                                          hidden_channels=dims,
                                          bn=True)

    @add_start_docstrings(ROBERTA_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        class_weights=None,
        cat_feats=None,
        numerical_feats=None
    ):
        r"""
        class_weights (:obj:`torch.FloatTensor` of shape :obj:`(tabular_config.num_labels,)`, `optional`, defaults to :obj:`None`):
            Class weights to be used for cross entropy loss function for classification task
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`tabular_config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`tabular_config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        cat_feats (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, tabular_config.cat_feat_dim)`, `optional`, defaults to :obj:`None`):
            Categorical features to be passed in to the TabularFeatCombiner
        numerical_feats (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, tabular_config.numerical_feat_dim)`, `optional`, defaults to :obj:`None`):
            Numerical features to be passed in to the TabularFeatCombiner

    Returns:
        :obj:`tuple` comprising various elements depending on configuration and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if tabular_config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, tabular_config.num_labels)`):
            Classification (or regression if tabular_config.num_labels==1) scores (before SoftMax).
        classifier_layer_outputs(:obj:`list` of :obj:`torch.FloatTensor`):
            The outputs of each layer of the final classification layers. The 0th index of this list is the
            combining module's output

        """
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]
        text_feats = sequence_output[:, 0, :]
        text_feats = self.dropout(text_feats)
        print('Sequence Outputs Shape')
        print(sequence_output.shape)
        print('Text Feats Shape')
        print(text_feats.shape)
        print('Cat Feats Shape')
        print(cat_feats.shape)
        combined_feats = self.tabular_combiner(text_feats,
                                               cat_feats,
                                               numerical_feats)
        loss, logits, classifier_layer_outputs = hf_loss_func(combined_feats,
                                                              self.tabular_classifier,
                                                              labels,
                                                              self.num_labels,
                                                              class_weights)
        return loss, logits, classifier_layer_outputs


class XLMRobertaWithTabular(RobertaWithTabular):
    """
        This class overrides :class:`~RobertaWithTabular`. Please check the
        superclass for the appropriate documentation alongside usage examples.
        """
    config_class = XLMRobertaConfig


class DistilBertWithTabular(DistilBertForSequenceClassification):
    """
    DistilBert Model transformer with a sequence classification/regression head as well as
    a TabularFeatCombiner module to combine categorical and numerical features
    with the Roberta pooled output

    Parameters:
        hf_model_config (:class:`~transformers.DistilBertConfig`):
            Model configuration class with all the parameters of the model.
            This object must also have a tabular_config member variable that is a
            :obj:`TabularConfig` instance specifying the configs for :obj:`TabularFeatCombiner`
    """
    def __init__(self, hf_model_config):
        super().__init__(hf_model_config)
        tabular_config = hf_model_config.tabular_config
        if type(tabular_config) is dict:  # when loading from saved model
            tabular_config = TabularConfig(**tabular_config)
        else:
            self.config.tabular_config = tabular_config.__dict__

        tabular_config.text_feat_dim = hf_model_config.hidden_size
        tabular_config.hidden_dropout_prob = hf_model_config.seq_classif_dropout
        self.tabular_combiner = TabularFeatCombiner(tabular_config)
        self.num_labels = tabular_config.num_labels
        combined_feat_dim = self.tabular_combiner.final_out_dim
        if tabular_config.use_simple_classifier:
            self.tabular_classifier = nn.Linear(combined_feat_dim,
                                                tabular_config.num_labels)
        else:
            dims = calc_mlp_dims(combined_feat_dim,
                                 division=tabular_config.mlp_division,
                                 output_dim=tabular_config.num_labels)
            self.tabular_classifier = MLP(combined_feat_dim,
                                          tabular_config.num_labels,
                                          num_hidden_lyr=len(dims),
                                          dropout_prob=tabular_config.mlp_dropout,
                                          hidden_channels=dims,
                                          bn=True)

    @add_start_docstrings(DISTILBERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        class_weights=None,
        cat_feats=None,
        numerical_feats=None
    ):
        r"""
        class_weights (:obj:`torch.FloatTensor` of shape :obj:`(tabular_config.num_labels,)`,`optional`, defaults to :obj:`None`):
            Class weights to be used for cross entropy loss function for classification task
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`tabular_config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`tabular_config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        cat_feats (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, tabular_config.cat_feat_dim)`,`optional`, defaults to :obj:`None`):
            Categorical features to be passed in to the TabularFeatCombiner
        numerical_feats (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, tabular_config.numerical_feat_dim)`,`optional`, defaults to :obj:`None`):
            Numerical features to be passed in to the TabularFeatCombiner
    Returns:
        :obj:`tuple` comprising various elements depending on configuration and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if tabular_config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, tabular_config.num_labels)`):
            Classification (or regression if tabular_config.num_labels==1) scores (before SoftMax).
        classifier_layer_outputs(:obj:`list` of :obj:`torch.FloatTensor`):
            The outputs of each layer of the final classification layers. The 0th index of this list is the
            combining module's output
        """

        distilbert_output = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        text_feats = self.dropout(pooled_output)
        combined_feats = self.tabular_combiner(text_feats,
                                               cat_feats,
                                               numerical_feats)
        loss, logits, classifier_layer_outputs = hf_loss_func(combined_feats,
                                                              self.tabular_classifier,
                                                              labels,
                                                              self.num_labels,
                                                              class_weights)
        return loss, logits, classifier_layer_outputs


class AlbertWithTabular(AlbertForSequenceClassification):
    """
    ALBERT Model transformer with a sequence classification/regression head as well as
    a TabularFeatCombiner module to combine categorical and numerical features
    with the Roberta pooled output

    Parameters:
        hf_model_config (:class:`~transformers.AlbertConfig`):
            Model configuration class with all the parameters of the model.
            This object must also have a tabular_config member variable that is a
            :obj:`TabularConfig` instance specifying the configs for :obj:`TabularFeatCombiner`
    """

    def __init__(self, hf_model_config):
        super().__init__(hf_model_config)
        tabular_config = hf_model_config.tabular_config
        if type(tabular_config) is dict:  # when loading from saved model
            tabular_config = TabularConfig(**tabular_config)
        else:
            self.config.tabular_config = tabular_config.__dict__

        tabular_config.text_feat_dim = hf_model_config.hidden_size
        tabular_config.hidden_dropout_prob = hf_model_config.hidden_dropout_prob
        self.tabular_combiner = TabularFeatCombiner(tabular_config)
        self.num_labels = tabular_config.num_labels
        combined_feat_dim = self.tabular_combiner.final_out_dim
        if tabular_config.use_simple_classifier:
            self.tabular_classifier = nn.Linear(combined_feat_dim,
                                                tabular_config.num_labels)
        else:
            dims = calc_mlp_dims(combined_feat_dim,
                                 division=tabular_config.mlp_division,
                                 output_dim=tabular_config.num_labels)
            self.tabular_classifier = MLP(combined_feat_dim,
                                          tabular_config.num_labels,
                                          num_hidden_lyr=len(dims),
                                          dropout_prob=tabular_config.mlp_dropout,
                                          hidden_channels=dims,
                                          bn=True)

    @add_start_docstrings(ALBERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        class_weights=None,
        cat_feats=None,
        numerical_feats=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        combined_feats = self.tabular_combiner(pooled_output,
                                               cat_feats,
                                               numerical_feats)
        loss, logits, classifier_layer_outputs = hf_loss_func(combined_feats,
                                                              self.tabular_classifier,
                                                              labels,
                                                              self.num_labels,
                                                              class_weights)
        return loss, logits, classifier_layer_outputs


class XLNetWithTabular(XLNetForSequenceClassification):
    """
    XLNet Model transformer with a sequence classification/regression head as well as
    a TabularFeatCombiner module to combine categorical and numerical features
    with the Roberta pooled output

    Parameters:
        hf_model_config (:class:`~transformers.XLNetConfig`):
            Model configuration class with all the parameters of the model.
            This object must also have a tabular_config member variable that is a
            :obj:`TabularConfig` instance specifying the configs for :obj:`TabularFeatCombiner`
    """
    def __init__(self, hf_model_config):
        super().__init__(hf_model_config)
        tabular_config = hf_model_config.tabular_config
        if type(tabular_config) is dict:  # when loading from saved model
            tabular_config = TabularConfig(**tabular_config)
        else:
            self.config.tabular_config = tabular_config.__dict__

        tabular_config.text_feat_dim = hf_model_config.hidden_size
        self.tabular_combiner = TabularFeatCombiner(tabular_config)
        self.num_labels = tabular_config.num_labels
        combined_feat_dim = self.tabular_combiner.final_out_dim
        if tabular_config.use_simple_classifier:
            self.tabular_classifier = nn.Linear(combined_feat_dim,
                                                tabular_config.num_labels)
        else:
            dims = calc_mlp_dims(combined_feat_dim,
                                 division=tabular_config.mlp_division,
                                 output_dim=tabular_config.num_labels)
            self.tabular_classifier = MLP(combined_feat_dim,
                                          tabular_config.num_labels,
                                          num_hidden_lyr=len(dims),
                                          dropout_prob=tabular_config.mlp_dropout,
                                          hidden_channels=dims,
                                          bn=True)

    @add_start_docstrings(XLNET_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        token_type_ids=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        class_weights=None,
        cat_feats=None,
        numerical_feats=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`)
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = self.training or (use_cache if use_cache is not None else self.config.use_cache)

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        output = transformer_outputs[0]

        output = self.sequence_summary(output)
        combined_feats = self.tabular_combiner(output,
                                               cat_feats,
                                               numerical_feats)
        loss, logits, classifier_layer_outputs = hf_loss_func(combined_feats,
                                                              self.tabular_classifier,
                                                              labels,
                                                              self.num_labels,
                                                              class_weights)
        return loss, logits, classifier_layer_outputs


class XLMWithTabular(XLMForSequenceClassification):
    """
    XLM Model transformer with a sequence classification/regression head as well as
    a TabularFeatCombiner module to combine categorical and numerical features
    with the Roberta pooled output

    Parameters:
        hf_model_config (:class:`~transformers.XLMConfig`):
            Model configuration class with all the parameters of the model.
            This object must also have a tabular_config member variable that is a
            :obj:`TabularConfig` instance specifying the configs for :obj:`TabularFeatCombiner`
    """
    def __init__(self, hf_model_config):
        super().__init__(hf_model_config)
        tabular_config = hf_model_config.tabular_config
        if type(tabular_config) is dict:  # when loading from saved model
            tabular_config = TabularConfig(**tabular_config)
        else:
            self.config.tabular_config = tabular_config.__dict__

        tabular_config.text_feat_dim = hf_model_config.hidden_size
        self.tabular_combiner = TabularFeatCombiner(tabular_config)
        self.num_labels = tabular_config.num_labels
        combined_feat_dim = self.tabular_combiner.final_out_dim
        if tabular_config.use_simple_classifier:
            self.tabular_classifier = nn.Linear(combined_feat_dim,
                                                tabular_config.num_labels)
        else:
            dims = calc_mlp_dims(combined_feat_dim,
                                 division=tabular_config.mlp_division,
                                 output_dim=tabular_config.num_labels)
            self.tabular_classifier = MLP(combined_feat_dim,
                                          tabular_config.num_labels,
                                          num_hidden_lyr=len(dims),
                                          dropout_prob=tabular_config.mlp_dropout,
                                          hidden_channels=dims,
                                          bn=True)

    @ add_start_docstrings(XLM_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            langs=None,
            token_type_ids=None,
            position_ids=None,
            lengths=None,
            cache=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            class_weights=None,
            cat_feats=None,
            numerical_feats=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        output = transformer_outputs[0]
        output = self.sequence_summary(output)
        combined_feats = self.tabular_combiner(output,
                                               cat_feats,
                                               numerical_feats)
        loss, logits, classifier_layer_outputs = hf_loss_func(combined_feats,
                                                              self.tabular_classifier,
                                                              labels,
                                                              self.num_labels,
                                                              class_weights)
        return loss, logits, classifier_layer_outputs

class LongformerWithTabular(LongformerForSequenceClassification):
    """
    Longformer Model With Sequence Classification Head
    """
    def __init__(self, hf_model_config):
        super().__init__(hf_model_config)
        tabular_config = hf_model_config.tabular_config
        if type(tabular_config) is dict:  # when loading from saved model
            tabular_config = TabularConfig(**tabular_config)
        else:
            self.config.tabular_config = tabular_config.__dict__

        self.add_attention_module = tabular_config.add_attention_module
        self.save_attentions = tabular_config.save_attentions
        self.attentions_path = tabular_config.attentions_path

        tabular_config.text_feat_dim = hf_model_config.hidden_size
        tabular_config.hidden_dropout_prob = hf_model_config.hidden_dropout_prob
        self.tabular_combiner = TabularFeatCombiner(tabular_config)
        self.num_labels = tabular_config.num_labels
        combined_feat_dim = self.tabular_combiner.final_out_dim
        self.dropout = nn.Dropout(hf_model_config.hidden_dropout_prob)
        self.batch_size = tabular_config.batch_size
        # print('Keyword MLP out dim is ', tabular_config.keyword_MLP_out_dim)

        if self.add_attention_module:
            print('Number of keywords is ', tabular_config.num_keywords)
            self.keyword_MLP = MLP(300 * tabular_config.num_keywords * 2, tabular_config.keyword_MLP_out_dim, num_hidden_lyr=1, dropout_prob=0.1, hidden_channels=[600], bn=True)

        if tabular_config.use_simple_classifier:
            if self.add_attention_module:
                print('Combined Feat Dim is ', combined_feat_dim)
                print('keyword MLP out dim is ', self.keyword_MLP.out_dim)
                self.tabular_classifier = nn.Linear(combined_feat_dim + self.keyword_MLP.out_dim - 100,
                                                    tabular_config.num_labels)
            else:
                self.tabular_classifier = nn.Linear(combined_feat_dim, tabular_config.num_labels)
        else:
            if self.add_attention_module:
                dims = calc_mlp_dims(combined_feat_dim + self.keyword_MLP.out_dim,
                                    division=tabular_config.mlp_division,
                                    output_dim=tabular_config.num_labels)
                self.tabular_classifier = MLP(combined_feat_dim + self.keyword_MLP.out_dim,
                                            tabular_config.num_labels,
                                            num_hidden_lyr=len(dims),
                                            dropout_prob=tabular_config.mlp_dropout,
                                            hidden_channels=dims,
                                            bn=True)
            else:
                dims = calc_mlp_dims(combined_feat_dim,
                                    division=tabular_config.mlp_division,
                                    output_dim=tabular_config.num_labels)
                self.tabular_classifier = MLP(combined_feat_dim,
                                            tabular_config.num_labels,
                                            num_hidden_lyr=len(dims),
                                            dropout_prob=tabular_config.mlp_dropout,
                                            hidden_channels=dims,
                                            bn=True)

        # dangerous harcoding, fix later
        if self.add_attention_module:
            print('Vocab size is ', tabular_config.vocab_size)

            # there was once an idiot named Sidharrth
            if tabular_config.group == 'practice-a':
                self.embedding_layer = nn.Embedding(num_embeddings=tabular_config.vocab_size, embedding_dim=300)
            else:
                self.embedding_layer = nn.Embedding(num_embeddings=tabular_config.vocab_size + 1, embedding_dim=300)


            self.att_layer = KeyAttention(
                name='attention',
                op='dp',
                seed=0,
                emb_dim=300,
                word_att_pool='mean',
                merge_ans_key='concat',
                beta=False,
                batch_size=self.batch_size,
                tabular_config=tabular_config
            )

        # load embeddings
        # self.embedding_layer = nn.Embedding(self.config.vocab_size, embedding_dim=300)
        # self.embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(tabular_config.embedding_weights).float(), freeze=True)
    @add_start_docstrings(LONGFORMER_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        head_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        class_weights=None,
        cat_feats=None,
        numerical_feats=None,
        answer_tokens=None,
        keyword_tokens=None,
        answer_mask=None,
        keyword_mask=None,
        lemmatized_answer_tokens=None,
    ):
        if global_attention_mask is None:
            # print("Initializing global attention on CLS token...")
            global_attention_mask = torch.zeros_like(input_ids)
            # global attention on cls token
            global_attention_mask[:, 0] = 1

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            # head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        text_feats = sequence_output[:, 0, :]
        text_feats = self.dropout(text_feats)
        # print('Sequence Outputs Shape')
        # print(sequence_output.shape)
        # print('Text Feats Shape')
        # print(text_feats.shape)
        # print('Cat Feats Shape')
        # print(cat_feats.shape)
        combined_feats = self.tabular_combiner(text_feats,
                                               cat_feats,
                                               numerical_feats,
                                            )

        if self.add_attention_module:
            ans_emb = self.embedding_layer(lemmatized_answer_tokens)
            ans_mask_emb = self.embedding_layer(answer_mask)
            keys_emb = self.embedding_layer(keyword_tokens)
            keys_mask_emb = self.embedding_layer(keyword_mask)

            att_rtn_keys = ('z', 'z_ans', 'z_key', 'beta_ans', 'beta_key')

            fea_att_list = list()
            attentions = {k: list() for k in att_rtn_keys}

            # was this causing CUDA error
            # first dimension is batch size
            for i in range(keyword_tokens.shape[1]):
                t_k = LambdaLayer(lambda x: x[:, i], name='key_%d' % i)(keys_emb)
                t_k_m = LambdaLayer(lambda x: x[:, i], name='ans_%d' % i)(keyword_mask)
                # t_k_m = LambdaLayer(lambda x: x[:, i], name='ans_%d' % i)(keyword_tokens)

                f, *att_rtn = self.att_layer([ans_emb, answer_mask, t_k, t_k_m])

                fea_att_list.append(f)

                for i_a_r, a_r in enumerate(att_rtn):
                    attentions[att_rtn_keys[i_a_r]].append(a_r)

            # do something with this- represents keyword attention
            # print('fea_att_list.shape', len(fea_att_list))
            # print('fea_att_list', fea_att_list)
            # print('Shapes of items in fea_att_list')
            # for item in fea_att_list:
            #     print(item.shape)
            fea_rubric = torch.cat(fea_att_list, dim=1)

            # print('fea_rubric shape', fea_rubric.shape)
            # print('fea_rubric', fea_rubric)

            layer_exp_dim = LambdaLayer(lambda x: torch.unsqueeze(x, 1), name='expand_dim')
            att_list = list()

            for k in attentions:
                att_k = attentions[k]
                att_exp = [layer_exp_dim(m) for m in att_k]
                if len(att_exp) == 1:
                    att_exp = att_exp[0]
                else:
                    att_exp = torch.cat(att_exp, dim=1)
                attentions[k] = att_exp
                att_list.append(att_exp)

            # print('Before loss')

            # fea_att_list = [combined_feats, fea_rubric]
            keyword_feats = self.keyword_MLP(fea_rubric.float())
            fea_att_list = torch.cat([combined_feats, keyword_feats], dim=1)
            loss, logits, classifier_layer_outputs = hf_loss_func(fea_att_list,
                                                                self.tabular_classifier,
                                                                labels,
                                                                self.num_labels,
                                                                class_weights)
            if self.save_attentions:
                with open(self.attentions_path, 'wb') as handle:
                    pickle.dump(attentions, handle)

            return loss, logits, classifier_layer_outputs

        else:
            loss, logits, classifier_layer_outputs = hf_loss_func(combined_feats,
                                                                self.tabular_classifier,
                                                                labels,
                                                                self.num_labels,
                                                                class_weights)
            return loss, logits, classifier_layer_outputs