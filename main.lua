require 'torch'
require 'image'
require 'nn'
dsg_utils = require 'dsg_utils'
dsg_nets = require 'dsg_nets'

local cuda_flag = false

-- Load training set
trainset = dsg_utils.LoadAndAugmentDataset("id_train.csv")

--for i = 1,math.min(ntrain,5) do
--    local pos = math.random(ntrain)
--    image.display{image=trainset.data[pos], legend=trainset.label[pos]}
--end

local net
local mean
local std
--net, mean, std = dsg_utils.KFoldedCV(trainset, 4, dsg_nets.Lenet, cuda_flag)
mean, stdv = dsg_utils.Normalize(trainset)
torch.save('model.mean', mean)
torch.save('model.stdv', stdv)

net = dsg_utils.TrainNet(trainset, dsg_nets.Lenet, 'heuristic', cuda_flag)
torch.save('model.net', net)
