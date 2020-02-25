import torch
import math
import numpy
import scipy.stats
import itertools
import statsmodels.stats.proportion
from scipy.special import comb
import random
def random_mask_batch_one_sample(batch, block_size , reuse_noise = False):
	batch = batch.permute(0,2,3,1) #color channel last
	out_c1 = torch.zeros(batch.shape).cuda()
	out_c2 = torch.zeros(batch.shape).cuda()
	if (reuse_noise):
		pos = random.randint(0, batch.shape[2]-1)
		if  (pos+block_size > batch.shape[2]):
			out_c1[:,:,pos:] = batch[:,:,pos:]
			out_c2[:,:,pos:] = 1. - batch[:,:,pos:]

			out_c1[:,:,:pos+block_size-batch.shape[2]] = batch[:,:,:pos+block_size-batch.shape[2]]
			out_c2[:,:,:pos+block_size-batch.shape[2]] = 1. - batch[:,:,:pos+block_size-batch.shape[2]]
		else:
			out_c1[:,:,pos:pos+block_size] = batch[:,:,pos:pos+block_size]
			out_c2[:,:,pos:pos+block_size] = 1. - batch[:,:,pos:pos+block_size]

	else:
		for i in range(batch.shape[0]):
			pos = random.randint(0, batch.shape[2]-1)
			if  (pos+block_size > batch.shape[2]):
				out_c1[i,:,pos:] = batch[i,:,pos:]
				out_c2[i,:,pos:] = 1. - batch[i,:,pos:]

				out_c1[i,:,:pos+block_size-batch.shape[2]] = batch[i,:,:pos+block_size-batch.shape[2]]
				out_c2[i,:,:pos+block_size-batch.shape[2]] = 1. - batch[i,:,:pos+block_size-batch.shape[2]]
			else:
				out_c1[i,:,pos:pos+block_size] = batch[i,:,pos:pos+block_size]
				out_c2[i,:,pos:pos+block_size] = 1. - batch[i,:,pos:pos+block_size]
	out_c1 = out_c1.permute(0,3,1,2)
	out_c2 = out_c2.permute(0,3,1,2)
	out = torch.cat((out_c1,out_c2), 1)
	#print(out[14,:,5:10,5:10])
	return out
def avg_hard_forward(batch, net, num_samples, block_size, sub_batch = 0):
	if (sub_batch == 0):
		expanded = batch.repeat_interleave(num_samples,0) # shape: batch*num_samples, etc
		masked = random_mask_batch_one_sample(expanded, block_size)
		soft = net(masked)
		votes = soft.max(1)[1]
		hard = torch.zeros(soft.shape).cuda()
		hard.scatter_(1,votes.unsqueeze(1),1)
		return hard.reshape((batch.shape[0],num_samples,) + hard.shape[1:]).mean(dim=1)
	else:
		aresums = False
		count = num_samples
		while (count > 0):
			curr_sub_batch = min(count, sub_batch)
			count -= sub_batch
			expanded = batch.repeat_interleave(curr_sub_batch,0) # shape: batch*num_samples, etc
			masked = random_mask_batch_one_sample(expanded, block_size)
			soft = net(masked)
			votes = soft.max(1)[1]
			hard = torch.zeros(soft.shape).cuda()
			hard.scatter_(1,votes.unsqueeze(1),1)
			if (aresums == False):
				aresums = True
				sums = torch.zeros((batch.shape[0],) +  hard.shape[1:]).cuda()
			sums += hard.reshape((batch.shape[0],curr_sub_batch,) + hard.shape[1:]).sum(dim=1)
		return sums/num_samples
def lc_bound(k, n ,alpha):
	return statsmodels.stats.proportion.proportion_confint(k, n, alpha=2*alpha, method="beta")[0]
# returns -1 for incorrect, 0 for  no certificate, 1 for correct certificate
def certify(batch, labels, net, alpha, block_size, size_to_certify, num_samples_select, num_samples_bound, sub_batch = 0 ):
	guesses = avg_hard_forward(batch, net, num_samples_select, block_size,sub_batch=sub_batch).max(1)[1]
	bound_scores = avg_hard_forward(batch, net, num_samples_bound, block_size,sub_batch=sub_batch)		
	bound_selected_scores = torch.gather(bound_scores,1,guesses.unsqueeze(1)).squeeze(0)
	if(len(bound_selected_scores.shape) == 1):
		bound_selected_scores = bound_selected_scores.unsqueeze(0)
	bound_selected_scores = lc_bound((bound_selected_scores*num_samples_bound).cpu().numpy(),num_samples_bound,alpha)
	threshold  = .5 + float(size_to_certify + block_size -1)/batch.shape[2]
	radii = torch.tensor(bound_selected_scores > threshold).type(torch.int)
	radii[guesses != labels] *= -1
	return radii
def predict(batch, net ,block_size, num_samples, alpha, sub_batch = 0 ):
	scores = avg_hard_forward(batch, net, num_samples, block_size,sub_batch=sub_batch)
	toptwo = torch.topk(scores.cpu(),2,sorted=True)
	toptwoidx = toptwo[1]
	toptwocounts = toptwo[0]*num_samples
	out = -1* torch.ones(batch.shape[0], dtype = torch.long)
	tests = numpy.array([scipy.stats.binom_test(toptwocounts[idx,0],toptwocounts[idx,0]+toptwocounts[idx,1], .5) for idx in range(batch.shape[0])])
	out[tests <= alpha] = toptwoidx[tests <= alpha][:,0]
	return out

def batch_choose(n,k,batches):
	#start = torch.cuda.Event(enable_timing=True)
	#end = torch.cuda.Event(enable_timing=True)
	#start.record()
	out = torch.zeros((batches,k), dtype=torch.long).cuda()
	for i in range(k):
		out[:,i] = torch.randint(0,n-i, (batches,))
		if (i != 0):
			last_boost = torch.zeros(batches, dtype=torch.long).cuda()
			boost = (out[:,:i] <=(out[:,i]+last_boost).unsqueeze(0).t()).sum(dim=1)
			while (boost.eq(last_boost).sum() != batches):
				last_boost = boost
				boost = (out[:,:i] <=(out[:,i]+last_boost).unsqueeze(0).t()).sum(dim=1)
			out[:,i]  += boost
	#end.record()
	#torch.cuda.synchronize()
	#print(start.elapsed_time(end))
	return out
