from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import numpy as np
import tensorflow as tf

import copy
import itertools
from six.moves import zip
from tensorflow import dtypes
from ultra.learning_algorithm.base_algorithm import BaseAlgorithm
import ultra.utils
from ultra.utils import losses_impl
from ultra.utils import utils_func

DEFAULT_GAIN_FN = lambda label: label
EXP_GAIN_FN = lambda label: tf.pow(2.0, label) - 1
DEFAULT_RANK_DISCOUNT_FN = lambda rank: tf.math.log(2.) / tf.math.log1p(rank)
RECIPROCAL_RANK_DISCOUNT_FN = lambda rank: 1. / rank

def create_dcg_lambda_weight(discount_func, gain_func, topn=None, normalized=True, smooth_fraction=0.):
    """Creates _LambdaWeight for DCG metric."""
    RANK_DISCOUNT_FN = DEFAULT_RANK_DISCOUNT_FN if discount_func == 'log1p' else RECIPROCAL_RANK_DISCOUNT_FN
    GAIN_FN = DEFAULT_GAIN_FN if gain_func == 'linear' else EXP_GAIN_FN
    return losses_impl.DCGLambdaWeight(
        topn,
        gain_fn=GAIN_FN,
        rank_discount_fn=RANK_DISCOUNT_FN,
        normalized=normalized,
        smooth_fraction=smooth_fraction)

def get_bernoulli_sample(probs):
    """Conduct Bernoulli sampling according to a specific probability distribution.

        Args:
            prob: (tf.Tensor) A tensor in which each element denotes a probability of 1 in a Bernoulli distribution.

        Returns:
            A Tensor of binary samples (0 or 1) with the same shape of probs.

        """
    return tf.ceil(probs - tf.random_uniform(tf.shape(probs)))


class PairwiseRegressionEM(BaseAlgorithm):
    """The Pairwise Regression EM algorithm for unbiased learning to rank.

    """

    def __init__(self, data_set, exp_settings, forward_only=False):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
            forward_only: Set true to conduct prediction only, false to conduct training.
        """
        print('Build Pairwise RegressionEM algorithm.')
        self.tau = 1e-12
        self.hparams = ultra.utils.hparams.HParams(
            EM_step_size=0.05,                  # Step size for EM algorithm.
            learning_rate=0.05,                 # Learning rate.
            step_decay_ratio=-0.05,
            max_gradient_norm=5.0,            # Clip gradients to this norm.
            enable_ips=True,
            pointwise_only=False,
            gain_fn='exp',
            discount_fn='log1p',
            opt_metric='ndcg',
            corr_pair_clk_noise=True,
            # Set strength for L2 regularization.
            l2_loss=0.00,
            grad_strategy='ada',            # Select gradient strategy
        )
        print(exp_settings['learning_algorithm_hparams'])
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        print("in pairwise regem.. hparams:", self.hparams)
        self.exp_settings = exp_settings
        self.model = None
        self.max_candidate_num = exp_settings['max_candidate_num']
        self.feature_size = data_set.feature_size
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.Variable(
            float(self.hparams.learning_rate), trainable=False) * \
            tf.math.pow((1+tf.cast(self.global_step, tf.float32)), \
                        self.hparams.step_decay_ratio)
        self.curr_EM_step_size = self.hparams.EM_step_size * \
            tf.math.pow((1+tf.cast(self.global_step, tf.float32)), \
                        self.hparams.step_decay_ratio)
        tf.summary.scalar(
            'curr_EM_step_size',
            self.curr_EM_step_size,
            collections=['train'])        
        self.lambda_weight = None
        if self.hparams.opt_metric == 'ndcg':
            self.lambda_weight = create_dcg_lambda_weight(self.hparams.discount_fn, \
                                                          self.hparams.gain_fn, \
                                                          normalized=True)
        elif self.hparams.opt_metric == 'dcg':
            self.lambda_weight = create_dcg_lambda_weight(self.hparams.discount_fn, \
                                                          self.hparams.gain_fn, \
                                                          normalized=False)
        print(self.lambda_weight)
        # Feeds for inputs.
        self.is_training = tf.placeholder(tf.bool, name="is_train")
        self.docid_inputs = []  # a list of top documents
        self.letor_features = tf.placeholder(tf.float32, shape=[None, self.feature_size],
                                             name="letor_features")  # the letor features for the documents
        self.labels = []  # the labels for the documents (e.g., clicks)
        for i in range(self.max_candidate_num):
            self.docid_inputs.append(tf.placeholder(tf.int64, shape=[None],
                                                    name="docid_input{0}".format(i)))
            self.labels.append(tf.placeholder(tf.float32, shape=[None],
                                              name="label{0}".format(i)))


        if self.hparams.pointwise_only:
            self.output = self.ranking_model(
                self.max_candidate_num, scope='point_ranking_model')
        elif self.hparams.enable_ips:
            self.output = self.ranking_model(
                self.max_candidate_num, scope='ips_ranking_model')              
        else:
            self.output = self.ranking_model(
                self.max_candidate_num, scope='ranking_model')            

        # reshape from [max_candidate_num, ?] to [?, max_candidate_num]
        reshaped_labels = tf.transpose(tf.convert_to_tensor(self.labels))
        pad_removed_output = self.remove_padding_for_metric_eval(
            self.docid_inputs, self.output)
        for metric in self.exp_settings['metrics']:
            for topn in self.exp_settings['metrics_topn']:
                metric_value = ultra.utils.make_ranking_metric_fn(
                    metric, topn)(reshaped_labels, pad_removed_output, None)
                tf.summary.scalar(
                    '%s_%d' %
                    (metric, topn), metric_value, collections=['eval'])

        # Build unbiased pairwise loss only when it is training
        if not forward_only:
            self.rank_list_size = exp_settings['selection_bias_cutoff']
            sigmoid_prob_b = tf.Variable(tf.ones([1]) - 1.0, name="sigmoid_prob_b")
            tf.summary.scalar(
                'sigmoid_prob_b',
                tf.reduce_mean(sigmoid_prob_b),
                collections=['train'])

            point_train_output = self.ranking_model(
                self.rank_list_size, scope='point_ranking_model')
            point_train_output = point_train_output + sigmoid_prob_b
            
            train_output = self.ranking_model(
                self.rank_list_size, scope='ranking_model')
            
            ips_train_output = self.ranking_model(
                self.rank_list_size, scope='ips_ranking_model')
            
            train_labels = self.labels[:self.rank_list_size]
            self.build_propensity_variables()                        

            # Conduct pointwise regression EM
            tf.summary.histogram("point_train_output", point_train_output, 
                collections=['train'])
            tf.summary.histogram("train_output", train_output, 
                collections=['train'])
            beta = tf.sigmoid(point_train_output)
            tf.summary.histogram("beta", beta, 
                collections=['train'])
            # reshape from [rank_list_size, ?] to [?, rank_list_size]
            reshaped_train_labels = tf.transpose(
                tf.convert_to_tensor(train_labels))
            tf.summary.histogram("reshaped_train_labels", reshaped_train_labels, 
                collections=['train'])
            binary_labels = tf.where(tf.math.greater(reshaped_train_labels,0.0), \
                x=tf.ones_like(reshaped_train_labels, dtype=tf.float32), \
                y=tf.zeros_like(reshaped_train_labels, dtype=tf.float32))
            tf.summary.histogram("binary_labels", binary_labels, 
                collections=['train'])
            self.pointwise_regression_EM(point_train_output, beta, binary_labels)
            self.loss = self.pointwise_loss
            self.maximization_op = self.pointwise_maximization_op
            # pairwise em step
            if not self.hparams.pointwise_only:
                self.pairwise_regression_EM(train_output, ips_train_output, \
                    reshaped_train_labels, beta, binary_labels)
                self.loss = self.pointwise_loss + self.pairwise_loss
                if self.hparams.enable_ips:
                    self.loss = self.loss + self.ips_loss
                self.maximization_op = tf.group([self.pointwise_maximization_op, \
                                                 self.pairwise_maximization_op])

            # Add l2 loss
            params = tf.trainable_variables()
            if self.hparams.l2_loss > 0:
                for p in params:
                    self.loss += self.hparams.l2_loss * tf.nn.l2_loss(p)

            # Select optimizer
            self.optimizer_func = tf.train.AdagradOptimizer
            if self.hparams.grad_strategy == 'sgd':
                self.optimizer_func = tf.train.GradientDescentOptimizer

            # Gradients and SGD update operation for training the model.
            opt = self.optimizer_func(self.hparams.learning_rate)
            self.gradients = tf.gradients(self.loss, params)
            for grad,var in zip(self.gradients, params):
                tf.summary.histogram(var.op.name, var, collections=['train'])
                if grad is not None:
                    tf.summary.histogram(var.op.name + '/gradients', 
                            grad, collections=['train'])
            #for var in tf.global_variables():
            #    tf.summary.histogram(var.op.name, var)

            if self.hparams.max_gradient_norm > 0:
                self.clipped_gradients, self.norm = tf.clip_by_global_norm(self.gradients,
                                                                           self.hparams.max_gradient_norm)
                self.updates = opt.apply_gradients(zip(self.clipped_gradients, params),
                                                   global_step=self.global_step)
                tf.summary.scalar(
                    'Gradient_Norm',
                    self.norm,
                    collections=['train'])
            else:
                self.norm = None
                self.updates = opt.apply_gradients(zip(self.gradients, params),
                                                   global_step=self.global_step)

            tf.summary.scalar(
                'Learning_Rate',
                self.learning_rate,
                collections=['train'])
            tf.summary.scalar(
                'Loss', tf.reduce_mean(
                    self.loss), collections=['train'])

            # reshape from [rank_list_size, ?] to [?, rank_list_size]
            reshaped_train_labels = tf.transpose(
                tf.convert_to_tensor(train_labels))
            pad_removed_train_output = self.remove_padding_for_metric_eval(
                self.docid_inputs, train_output)
            for metric in self.exp_settings['metrics']:
                for topn in self.exp_settings['metrics_topn']:
                    metric_value = ultra.utils.make_ranking_metric_fn(metric, topn)(
                        reshaped_train_labels, pad_removed_train_output, None)
                    tf.summary.scalar(
                        '%s_%d' %
                        (metric, topn), metric_value, collections=['train'])

        self.train_summary = tf.summary.merge_all(key='train')
        self.eval_summary = tf.summary.merge_all(key='eval')
        self.saver = tf.train.Saver(tf.global_variables())

    def pairwise_regression_EM(self, train_output, ips_train_output, \
            train_labels, beta, binary_labels):
        # Conduct pairwise expectation step
        train_labels_j = tf.expand_dims(train_labels, axis=1)
        train_labels_i = tf.expand_dims(train_labels, axis=-1)
        binary_labels_j = tf.expand_dims(binary_labels, axis=1)
        # S_ij = tf.sign(train_labels_i - train_labels_j)
        # S_ij = tf.where(tf.equal(S_ij, 0.0), x=-tf.ones_like(S_ij), y=S_ij)
        # pairwise_labels = (1 / 2) * (1 + S_ij)
        delta_labels = train_labels_i - train_labels_j
        tf.summary.histogram("pairwise_delta_labels", delta_labels, 
                collections=['train'])
        pairwise_labels = tf.where(tf.math.greater(delta_labels,0.0), \
            x=tf.ones_like(delta_labels, dtype=tf.float32), \
            y=tf.zeros_like(delta_labels, dtype=tf.float32))
        tf.summary.histogram("pairwise_binary_labels", pairwise_labels, 
                collections=['train'])

        train_output_j = tf.expand_dims(train_output, axis=1)
        train_output_i = tf.expand_dims(train_output, axis=-1)
        pairwise_logits = train_output_i - train_output_j
        tf.summary.histogram("pairwise_logits", pairwise_logits, 
                collections=['train'])

        theta_i = tf.expand_dims(self.propensity, axis=-1)
        theta_j = tf.expand_dims(self.propensity, axis=1)
        theta_minus_j = tf.expand_dims(self.propensity_minus, axis=1)
        theta_ij = theta_i * theta_j

        gamma = tf.sigmoid(pairwise_logits)
        tf.summary.histogram("gamma", gamma, 
                collections=['train'])
        beta_i = tf.expand_dims(beta, axis=-1)

        p_e11_r1_c1 = self.epsilon_plus * theta_ij * gamma
        p_e11_r0_c1 = self.epsilon_minus * theta_ij * (1.0 - gamma)
        p_e10_r1_c1 = theta_i * (1 - theta_j) * beta_i
        p_e10_r0_c1 = 0.0

        pos_prob = p_e11_r1_c1 + p_e11_r0_c1 + p_e10_r1_c1 + p_e10_r0_c1
        p_e11_r1_c1 = p_e11_r1_c1 / (pos_prob + self.tau)
        p_e11_r0_c1 = p_e11_r0_c1 / (pos_prob + self.tau)
        p_e10_r1_c1 = p_e10_r1_c1 / (pos_prob + self.tau)
        p_e10_r0_c1 = p_e10_r0_c1 / (pos_prob + self.tau)

        neg_prob = 1 - pos_prob
        p_e11_r1_c0 = (1 - self.epsilon_plus) * theta_ij * gamma
        p_e11_r0_c0 = (1 - self.epsilon_minus) * theta_ij * (1 - gamma)
        p_e11_r1_c0 = p_e11_r1_c0 / (neg_prob + self.tau)
        p_e11_r0_c0 = p_e11_r0_c0 / (neg_prob + self.tau)

        # conduct maximization step
        sum_p_e11_r1_c1 =  tf.reduce_sum(pairwise_labels * p_e11_r1_c1, axis=0,\
                 keep_dims=True) 
        sum_p_e11_r1_c0 = tf.reduce_sum((1-pairwise_labels) * p_e11_r1_c0, axis=0,\
                 keep_dims=True) 
        
        em_step_size = self.curr_EM_step_size
        self.update_epsilon_plus_op = self.epsilon_plus.assign(
                (1 - em_step_size) * self.epsilon_plus + \
                    em_step_size * \
                    (sum_p_e11_r1_c1)/(sum_p_e11_r1_c1+sum_p_e11_r1_c0+self.tau)
            )

        sum_p_e11_r0_c1 =  tf.reduce_sum(pairwise_labels * p_e11_r0_c1, axis=0,\
                 keep_dims=True) 
        sum_p_e11_r0_c0 = tf.reduce_sum((1-pairwise_labels) * p_e11_r0_c0, axis=0,\
                 keep_dims=True) 

        self.update_epsilon_minus_op = self.epsilon_minus.assign(
                (1 - em_step_size) * self.epsilon_minus + \
                    em_step_size * \
                    (sum_p_e11_r0_c1)/(sum_p_e11_r0_c1+sum_p_e11_r0_c0+self.tau)
            )       
        
        p_gamma_pos = self.epsilon_plus * gamma / \
                (self.epsilon_plus * gamma + self.epsilon_minus * (1 - gamma) + self.tau)

        denominator = theta_minus_j * self.epsilon_plus * gamma + \
                      theta_minus_j * self.epsilon_minus * (1 - gamma) + \
                      (1 - theta_minus_j) * beta_i 
        numerator =  theta_minus_j * self.epsilon_plus * gamma + \
                     ( \
                       (1 - theta_minus_j) * beta_i \
                      ) * 0.5
        p_gamma_neg = numerator / (denominator+self.tau)

        p_gamma = binary_labels_j * p_gamma_pos + (1 - binary_labels_j) * p_gamma_neg
        tf.summary.histogram("p_gamma", p_gamma, 
                collections=['train'])
        pairwise_ranker_labels = get_bernoulli_sample(p_gamma)
        tf.summary.histogram("pairwise_ranker_labels", pairwise_ranker_labels, 
                collections=['train'])
        
        mask = utils_func.is_label_valid(train_labels)
        ranks = losses_impl._compute_ranks(train_output, mask)

        if self.lambda_weight is not None:
            pairwise_weights = self.lambda_weight.pair_weights(train_labels, ranks)
            pairwise_weights *= tf.cast(tf.shape(input=train_labels)[1], dtype=tf.float32)

            pairwise_weights = tf.stop_gradient(
                pairwise_weights, name='weights_stop_gradient')
        else:
            pairwise_weights = 1.0

        losses = pairwise_labels * tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=pairwise_ranker_labels, logits=pairwise_logits)

        losses = losses * pairwise_weights          

        self.pairwise_loss = tf.reduce_mean(
                                    tf.reduce_sum(
                                        losses,
                                        axis=[1,2]
                                    )
                                )

        ips_train_output_j = tf.expand_dims(ips_train_output, axis=1)
        ips_train_output_i = tf.expand_dims(ips_train_output, axis=-1)
        ips_pairwise_logits = ips_train_output_i - ips_train_output_j
        tf.summary.histogram("ips_pairwise_logits", ips_pairwise_logits, 
                collections=['train'])
        m_ij = self.epsilon_plus / (self.epsilon_plus + self.epsilon_minus + self.tau)
        ips_weights1 = m_ij / (theta_ij + self.tau)
        ips_weights2 = ips_weights1 * theta_minus_j
        ips_weights = binary_labels_j * ips_weights1 + (1 - binary_labels_j) * ips_weights2

        losses = pairwise_labels * tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=pairwise_labels, logits=ips_pairwise_logits)
        losses = losses * pairwise_weights * ips_weights
        self.ips_loss = tf.reduce_mean(
                                    tf.reduce_sum(
                                        losses,
                                        axis=[1,2]
                                    )
                                )

        if self.hparams.corr_pair_clk_noise:
            self.pairwise_maximization_op = tf.group([self.update_epsilon_plus_op, \
                                                      self.update_epsilon_minus_op])
        else:
            self.pairwise_maximization_op = tf.no_op()

    def pointwise_regression_EM(self, train_output, beta, binary_labels):
        # Conduct pointwise expectation step
        p_e1_r1_c1 = 1.0
        p_e1_r0_c1 = 0.0       
        p_e0_c1 = 0.0
        neg_prob = 1 - (self.propensity * beta) + self.tau
        p_e1_r1_c0 = 0.0
        p_e1_r0_c0 = self.propensity * (1 - beta) / neg_prob
        p_e0_r1_c0 = (1 - self.propensity) * beta / neg_prob
        p_e0_r0_c0 = (1 - self.propensity) * (1 - beta) / neg_prob

        pos_prob = 1.0 - neg_prob

        # Conduct pointwise maximization step
        p_e1 = (1 - binary_labels) * (p_e1_r1_c0 + p_e1_r0_c0) \
                + binary_labels
        p_e1_c0 = (1 - binary_labels) * (p_e1_r1_c0 + p_e1_r0_c0)
        p_r1 = binary_labels * p_e1_r1_c1 + \
                (1 - binary_labels) * (p_e0_r1_c0 + p_e1_r1_c0)
        self.update_propensity_op = self.propensity.assign(
                (1 - self.curr_EM_step_size) * self.propensity + self.curr_EM_step_size * tf.reduce_mean(
                    p_e1, axis=0, keep_dims=True
                )
            )
            
        self.update_propensity_minus_op = self.propensity_minus.assign(
                (1 - self.curr_EM_step_size) * self.propensity_minus + \
                    self.curr_EM_step_size * \
                    tf.reduce_sum(p_e1_c0, axis=0, keep_dims=True) / \
                        (tf.reduce_sum(1-binary_labels, axis=0, keep_dims=True)+self.tau)
            )

        tf.summary.histogram(
                    'p_relevance',
                    p_r1,
                    collections=['train'])
        pointwise_ranker_labels = get_bernoulli_sample(p_r1)
        losses = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=pointwise_ranker_labels, logits=train_output)

        self.pointwise_loss = tf.reduce_mean(
                tf.reduce_sum(
                    losses,
                    axis=1
                )
            )
        self.pointwise_maximization_op = tf.group([self.update_propensity_op, \
                                                   self.update_propensity_minus_op])

    def build_propensity_variables(self):
        # Build propensity parameters
        self.propensity = tf.Variable(
                tf.ones([1, self.rank_list_size]) * 0.9, trainable=False)
        self.propensity_minus = tf.Variable(
                tf.ones([1, self.rank_list_size]) * 0.2, trainable=False)     

        if self.hparams.corr_pair_clk_noise:
            self.epsilon_plus = tf.Variable(
                    tf.ones([1, self.rank_list_size, self.rank_list_size]) * 0.95, trainable=False)
            self.epsilon_minus = tf.Variable(
                    tf.ones([1, self.rank_list_size, self.rank_list_size]) * 0.05, trainable=False)  
        else:
            self.epsilon_plus = tf.Variable(
                    tf.ones([1, self.rank_list_size, self.rank_list_size]) * 1.0, trainable=False)
            self.epsilon_minus = tf.Variable(
                    tf.ones([1, self.rank_list_size, self.rank_list_size]) * 0.0, trainable=False)            

        self.splitted_propensity = tf.split(
                self.propensity, self.rank_list_size, axis=1)
        self.splitted_propensity_minus = tf.split(
                self.propensity_minus, self.rank_list_size, axis=1)    

        self.splitted_epsilon_plus = tf.split(
                self.epsilon_plus, self.rank_list_size, axis=1)
        self.splitted_epsilon_minus = tf.split(
                self.epsilon_minus, self.rank_list_size, axis=1)

        for i in range(len(self.splitted_epsilon_plus)):
            self.splitted_epsilon_plus[i] = tf.split(self.splitted_epsilon_plus[i], 
                    self.rank_list_size, axis=2)
            self.splitted_epsilon_minus[i] = tf.split(self.splitted_epsilon_minus[i], 
                    self.rank_list_size, axis=2)

        for i in range(self.rank_list_size):
            tf.summary.scalar(
                    'splitted_propensity_%d' %
                    i,
                    tf.reduce_max(
                        self.splitted_propensity[i]),
                    collections=['train'])

            tf.summary.scalar(
                    'splitted_propensity_minus_%d' %
                    i,
                    tf.reduce_max(
                        self.splitted_propensity_minus[i]),
                    collections=['train'])

            for j in range(self.rank_list_size):
                tf.summary.scalar(
                        'splitted_epsilon_plus_%d_%d' % (i,j),
                        tf.reduce_max(
                            self.splitted_epsilon_plus[i][j]),
                        collections=['train'])
                tf.summary.scalar(
                        'splitted_epsilon_minus_%d_%d' % (i,j),
                        tf.reduce_max(
                            self.splitted_epsilon_minus[i][j]),
                        collections=['train'])

    def step(self, session, input_feed, forward_only):
        """Run a step of the model feeding the given inputs.

        Args:
            session: (tf.Session) tensorflow session to use.
            input_feed: (dictionary) A dictionary containing all the input feed data.
            forward_only: whether to do the backward step (False) or only forward (True).

        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a tf.summary containing related information about the step.

        """

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            input_feed[self.is_training.name] = True
            output_feed = [
                self.updates,    # Update Op that does SGD.
                self.loss,    # Loss for this batch.
                self.maximization_op,
                self.train_summary  # Summarize statistics.
            ]
        else:
            input_feed[self.is_training.name] = False
            output_feed = [
                self.eval_summary,  # Summarize statistics.
                self.output   # Model outputs
            ]

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            # loss, no outputs, summary.
            return outputs[1], None, outputs[-1]
        else:
            return None, outputs[1], outputs[0]    # loss, outputs, summary.
