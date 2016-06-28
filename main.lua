require 'torch'
require 'image'
require 'nn'
dsg_utils = require 'dsg_utils'
dsg_nets = require 'dsg_nets'

local cuda_flag = false
local model_name = 'model'

-- Load training set
--dsg_utils.PreprocessAndAugmentDataset("id_train.csv", "dsg_train_augmented.t7", "rgb")
trainset = torch.load("dsg_train_augmented.t7")

--net, mean, std = dsg_utils.KFoldedCV(trainset, 4, dsg_nets.Lenet, cuda_flag)
mean, stdv = dsg_utils.Normalize(trainset)
torch.save(model_name .. '.mean', mean)
torch.save(model_name .. '.stdv', stdv)

net = dsg_utils.TrainNet(trainset, dsg_nets.Lenet, 'heuristic', cuda_flag)
-- net = dsg_utils.TrainWithMinibatch(trainset, dsg_nets.VggBNDrop, 'msr', 50, 128, model_name, cuda_flag)
torch.save(model_name .. '.net', net)
