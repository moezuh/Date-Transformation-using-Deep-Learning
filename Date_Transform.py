# seq2seq on date transform
# learn to convert dates in the form "2021 March 5" (source sequence) to "3/5/2021" (target sequence)

import torch
import torch.nn as nn

# each character will be represented with a one-hot vector with a 1 at the position 
# specified by sourcedict (for source sequences) or targetdict (for target sequences)
sourcetub = [('0', 0), ('1', 1), ('2', 2), ('3', 3), ('4', 4), 
('5', 5), ('6', 6), ('7', 7), ('8', 8), ('9', 9), (' ', 10),
('A', 11), ('a', 11), ('b', 12), ('c', 13), ('D', 14), ('e', 15),
('F', 16), ('g', 17), ('h', 18), ('i', 19), ('J', 20), ('l', 21),
('M', 22), ('m', 22), ('N', 23), ('n', 23), ('O', 24), ('o', 24),
('p', 25), ('r', 26), ('S', 27), ('s', 27), ('t', 28), ('u', 29),
('v', 30), ('y', 31)]
sourcedict = dict(sourcetub)
targettub = [('0', 0), ('1', 1), ('2', 2), ('3', 3), ('4', 4), 
('5', 5), ('6', 6), ('7', 7), ('8', 8), ('9', 9), ('/', 10)]
targetdict = dict(targettub)
targetalphabet = [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '/']

lines = list(open("sampledates.txt", "r"))
source = []
target = []
for line in lines:
	parts = line.split("\t")
	source.append(parts[0])
	target.append(parts[1].strip())

batchsize = 80
sourcesize = 32
targetsize = 11
hiddensize = 40
seqlen1 = 18
seqlen2 = 11

torch.set_default_tensor_type(torch.DoubleTensor)
sourceinput = torch.zeros((seqlen1, batchsize, sourcesize))
for i in range(batchsize):
	l = [sourcedict[c] for c in source[i]]
	pos = 0
	for value in l:
		sourceinput[pos][i][value] = 1.0
		pos = pos + 1


targetinput = torch.zeros((seqlen2, batchsize, targetsize))
for i in range(batchsize):
	l = [targetdict[c] for c in target[i]]
	pos = 0
	for value in l:
		targetinput[pos][i][value] = 1.0
		pos = pos + 1

rnn1 = nn.RNNCell(sourcesize, hiddensize)
rnn2 = nn.RNNCell(targetsize, hiddensize)
hx1 = torch.randn(batchsize, hiddensize, requires_grad=False)
loss_fun = nn.CrossEntropyLoss()

eta = 0.1
n_iterations = 100 # Increase number of iterations for more accuracy
loss_fun = nn.CrossEntropyLoss()
for j in range(n_iterations):
    rnn1.zero_grad()
    rnn2.zero_grad()
    for i in range(seqlen1):
    	hx1 = rnn1(sourceinput[i], hx1)
    targetout = []
    loss = 0
    for i in range(seqlen2):
        hx2 = rnn2(targetinput[i], hx1)
        a = torch.argmax(targetinput[i], dim = 1)
        loss += loss_fun(hx2[:,0:targetsize], a)
        if j == n_iterations - 1:
            targetout.append(hx2[:,0:targetsize])
    print(j, loss.item()) 
    loss.backward(retain_graph = True)
    
    rnn1.weight_ih.data = rnn1.weight_ih.data - eta * rnn1.weight_ih.grad
    rnn1.bias_ih.data = rnn1.bias_ih.data - eta * rnn1.bias_ih.grad
    rnn1.weight_hh.data = rnn1.weight_hh.data - eta * rnn1.weight_hh.grad
    rnn1.bias_hh.data = rnn1.bias_hh.data - eta * rnn1.bias_hh.grad
    
    rnn2.weight_ih.data = rnn2.weight_ih.data - eta * rnn2.weight_ih.grad
    rnn2.bias_ih.data = rnn2.bias_ih.data - eta * rnn2.bias_ih.grad
    rnn2.weight_hh.data = rnn2.weight_hh.data - eta * rnn2.weight_hh.grad
    rnn2.bias_hh.data = rnn2.bias_hh.data - eta * rnn2.bias_hh.grad

for i in range(batchsize):
        output = ""
        for j in range(seqlen2):
            output = output + targetalphabet[torch.argmax(targetout[j][i]).item()]
        print(output, target[i])