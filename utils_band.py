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

previous_input_shape = None
previous_block_size = 0
previous_mask = None
def universal_mask(input_shape, block_size):
	global previous_input_shape
	global previous_block_size
	global previous_mask
	if (previous_mask is None or not (input_shape == previous_input_shape and previous_block_size == block_size)):
		print('Re-computing mask.....')
		permuted_shape = (input_shape[0],input_shape[2],input_shape[3],input_shape[1],)
		expanded_shape = (permuted_shape[0],permuted_shape[2],permuted_shape[1],permuted_shape[2],permuted_shape[3],)
		out= torch.zeros(expanded_shape, device='cuda')
		for pos in range(permuted_shape[2]):
			if  (pos+block_size > permuted_shape[2]):
				out[:,pos,:,pos:] = 1

				out[:,pos,:,:pos+block_size-permuted_shape[2]] = 1
			else:
				out[:,pos,:,pos:pos+block_size] =1
		out = out.reshape((permuted_shape[0]*permuted_shape[2],permuted_shape[1],permuted_shape[2],permuted_shape[3],))
		previous_mask = out.detach()
		previous_block_size = block_size
		previous_input_shape = input_shape
	return previous_mask


def forward_soft_parallel(inpt,net,block_size,num_classes,threshhold):
	predictions = torch.zeros(inpt.size(0), num_classes, device='cuda')
	mask = universal_mask(inpt.shape, block_size)

	batch = inpt.permute(0,2,3,1) #color channel last
	batch_view = batch.unsqueeze(1).expand(-1,batch.shape[2],-1,-1,-1).reshape(batch.shape[0]*batch.shape[2],batch.shape[1],batch.shape[2],batch.shape[3])
	#out_c1 = mask * batch_view
	#out_c2 = mask * (1-batch_view)
	out_c1=torch.zeros(mask.shape, device='cuda')
	out_c2=torch.zeros(mask.shape, device='cuda')
	nz = mask.nonzero(as_tuple=True)
	out_c1[nz] = batch_view[nz]
	out_c2[nz] = 1.-batch_view[nz]
	out_c1 = out_c1.permute(0,3,1,2)
	out_c2 = out_c2.permute(0,3,1,2)
	out = torch.cat((out_c1,out_c2), 1)
	softmx = torch.nn.functional.softmax(net(out),dim=1)
	softmx = softmx.reshape(batch.shape[0],batch.shape[2],num_classes)
	softout = softmx.mean(dim=1)
	hardout =  (softmx >= threshhold).type(torch.float).sum(dim=1)
	predinctionsnp = hardout.cpu().numpy()
	idxsort = numpy.argsort(-predinctionsnp,axis=1,kind='stable')
	hardclass = torch.tensor(idxsort[:,0]).cuda()
	return softout,hardclass


def predict_and_certify(inpt, net,block_size, size_to_certify, num_classes, threshold=0.0):
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
		softmx = torch.nn.functional.softmax(net(out),dim=1)
		#thresh, predicted = torch.nn.functional.softmax(net(out),dim=1).max(1)
		#print(thresh)
		predictions += (softmx >= threshold).type(torch.int).cuda()
	predinctionsnp = predictions.cpu().numpy()
	idxsort = numpy.argsort(-predinctionsnp,axis=1,kind='stable')
	valsort = -numpy.sort(-predinctionsnp,axis=1,kind='stable')
	val =  valsort[:,0]
	idx = idxsort[:,0]
	valsecond =  valsort[:,1]
	idxsecond =  idxsort[:,1] 
	num_affected_classifications=(size_to_certify + block_size -1)
	cert = torch.tensor(((val - valsecond >2*num_affected_classifications) | ((val - valsecond ==2*num_affected_classifications)&(idx < idxsecond)))).cuda()
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
