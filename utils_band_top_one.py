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
def predict_and_certify(inpt, net,block_size, size_to_certify, num_classes):
	predictions = torch.zeros(inpt.size(0), num_classes).type(torch.int).cuda()
	batch = inpt.permute(0,2,3,1) #color channel last
	for pos in range(batch.shape[2]):

		out_c1 = torch.zeros(batch.shape).cuda()
		out_c2 = torch.zeros(batch.shape).cuda()
		if  (pos+block_size > batch.shape[2]):
			out_c1[:,:,pos:] = batch[:,:,pos:]
			out_c2[:,:,pos:] = 1. - batch[:,:,pos:]

			out_c1[:,:,:pos+block_size-batch.shape[2]] = batch[:,:,:pos+block_size-batch.shape[2]]
			out_c2[:,:,:pos+block_size-batch.shape[2]] = 1. - batch[:,:,:pos+block_size-batch.shape[2]]
		else:
			out_c1[:,:,pos:pos+block_size] = batch[:,:,pos:pos+block_size]
			out_c2[:,:,pos:pos+block_size] = 1. - batch[:,:,pos:pos+block_size]

		out_c1 = out_c1.permute(0,3,1,2)
		out_c2 = out_c2.permute(0,3,1,2)
		out = torch.cat((out_c1,out_c2), 1)
		tsc, predicted = torch.nn.functional.softmax(net(out),dim=1).max(1)
		predictions[(torch.arange(inpt.size(0)),predicted)] += 1
	predinctionsnp = predictions.cpu().numpy()
	idxsort = numpy.argsort(-predinctionsnp,axis=1,kind='stable')
	valsort = -numpy.sort(-predinctionsnp,axis=1,kind='stable')
	val =  valsort[:,0]
	idx = idxsort[:,0]
	valsecond =  valsort[:,1]
	idxsecond =  idxsort[:,1] 
	num_affected_classifications=(size_to_certify + block_size -1)
	cert = torch.tensor((val - valsecond >2*num_affected_classifications) | ((val - valsecond ==2*num_affected_classifications)&(idx < idxsecond))).cuda()
	return torch.tensor(idx).cuda(), cert
#binom test(nA, nA + nB, p)

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
