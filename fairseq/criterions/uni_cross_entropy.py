# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from . import FairseqCriterion, register_criterion
from fairseq import utils

@register_criterion('uni_cross_entropy')
class UniCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, src_dict, dst_dict):
        super().__init__(args, src_dict, dst_dict)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss, as a Variable
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_input = sample['net_input']
        src_tokens = net_input['src_tokens']
        start = Variable(src_tokens.data[:,0:1].clone().fill_(self.eos_idx))
        net_input['prev_src_tokens'] = torch.cat((start, src_tokens[:,:-1]), dim=1)
        net_input['tgt_tokens'] = Variable(sample['target'].data.clone())
        net_input['sent_ids'] = sample['id']
        src_output,tgt_output,reg_loss = model(**net_input)

        lprobs = model.get_normalized_probs(tgt_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = sample['target'].view(-1)
        tgt_loss = F.nll_loss(lprobs, target, size_average=False, ignore_index=self.padding_idx, reduce=reduce)

        lprobs = model.get_normalized_probs(src_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = src_tokens.view(-1)
        src_loss = F.nll_loss(lprobs, target, size_average=False, ignore_index=self.padding_idx, reduce=reduce)

        loss = 0.5 * src_loss + 0.5 * tgt_loss + 100 * reg_loss
        
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens'] 
        src_sample_size = sample['source'].size(0) if self.args.sentence_avg else sample['src_ntokens'] 
        logging_output = {
            'loss': utils.item(tgt_loss.data) if reduce else tgt_loss.data,
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
            'src_loss': utils.item(src_loss.data) if reduce else src_loss.data,
            'src_ntokens': sample['src_ntokens'],
            'src_sample_size': src_sample_size,
            'reg_loss': utils.item(reg_loss.data) if reduce else reg_loss.data,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        src_loss_sum = sum(log.get('src_loss', 0) for log in logging_outputs)
        reg_loss_sum = sum(log.get('reg_loss', 0) for log in logging_outputs)

        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        src_ntokens = sum(log.get('src_ntokens', 0) for log in logging_outputs)

        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        src_sample_size = sum(log.get('src_sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'src_loss': src_loss_sum / src_sample_size / math.log(2),
            'reg_loss': reg_loss_sum
        }

        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
            agg_output['src_nll_loss'] = src_loss_sum / src_ntokens / math.log(2)

        return agg_output
