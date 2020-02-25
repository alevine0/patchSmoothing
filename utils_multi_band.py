import torch
import math
import numpy
import scipy.stats
import itertools
import statsmodels.stats.proportion
from scipy.special import comb
def random_mask_batch_one_sample(batch, keep_per_image, bandwidth, reuse_noise = False):
	batch = batch.permute(0,2,3,1) #color channel last
	out_c1 = torch.zeros(batch.shape).cuda()
	out_c2 = torch.zeros(batch.shape).cuda()
	if (reuse_noise):
		#ones = torch.ones(flat.shape[1]).cuda()
		#idx = torch.multinomial(ones,keep_per_image)
		idx = torch.tensor(numpy.random.choice(int(batch.shape[2]/bandwidth),keep_per_image,replace=False)).cuda()
		idx *= bandwidth
		for i in range(bandwidth):
			out_c1[:,:,idx+i] = batch[:,:,idx+i]
			out_c2[:,:,idx+i] = 1.-batch[:,:,idx+i]
	else:
		raise Error('Not implemented')
	out_c1 = out_c1.permute(0,3,1,2)
	out_c2 = out_c2.permute(0,3,1,2)
	out = torch.cat((out_c1,out_c2), 1)
	#print(out[14,:,5:10,5:10])
	return out
def predict_and_certify(inpt, net,keep, bandwidth, size_to_certify, num_classes, threshold=0.0):
	num_blocks = int(inpt.shape[3]/bandwidth)
	predictions = torch.zeros(inpt.size(0), num_classes).type(torch.int).cuda()
	for choices in itertools.combinations(range(num_blocks),keep):
		batch = inpt.permute(0,2,3,1) #color channel last
		out_c1 = torch.zeros(batch.shape).cuda()
		out_c2 = torch.zeros(batch.shape).cuda()
		idx = torch.tensor(choices).cuda()
		idx *= bandwidth
		for i in range(bandwidth):
			out_c1[:,:,idx+i] = batch[:,:,idx+i]
			out_c2[:,:,idx+i] = 1.-batch[:,:,idx+i]
		out_c1 = out_c1.permute(0,3,1,2)
		out_c2 = out_c2.permute(0,3,1,2)
		out = torch.cat((out_c1,out_c2), 1)
		thresh, predicted = torch.nn.functional.softmax(net(out),dim=1).max(1)
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
	num_affected_blocks= (int(math.ceil((size_to_certify-1)/bandwidth)) +1)
	num_affected_classifications  = comb(num_blocks, keep, exact=True) - comb(num_blocks-num_affected_blocks, keep, exact=True)
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
