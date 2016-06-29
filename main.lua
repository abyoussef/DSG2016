require 'torch'
require 'image'
require 'nn'
dsg_utils = require 'dsg_utils'
dsg_nets = require 'dsg_nets'

cmd = torch.CmdLine()
cmd:addTime()
cmd:option('-modelName', 'model', 'name of the model')
cmd:option('-preprocess', false, 'if true preprocess data')
cmd:option('-augmentedData', false, 'if true use augmented data')
cmd:option('-minibatch', false, 'if true train with minibatches')
cmd:option('-batchSize', 128, 'size of batches')
cmd:option('-nEpochs', 100, 'number of epochs')
cmd:option('-epochLearningStep', 25, 'number of epochs between learning rate change')
cmd:option('-epochSaveStep', 50, 'number of epochs between model save')
cmd:option('-cuda', false, 'if true train with minibatches')

opt = cmd:parse(arg or {})

-- Load training set
if opt.preprocess then
    if opt.augmentedData then
        dsg_utils.PreprocessAndAugmentDataset("id_train.csv", "dsg_train_augmented.t7", "rgb")
    else
        dsg_utils.PreprocessDataset("id_train.csv", "dsg_train.t7", "rgb")
    end
end
if opt.augmentedData then
    trainset = torch.load("dsg_train_augmented.t7")
else
    trainset = torch.load("dsg_train.t7")
end

--net, mean, std = dsg_utils.KFoldedCV(trainset, 4, dsg_nets.Lenet, opt.cuda)
mean, stdv = dsg_utils.Normalize(trainset)
torch.save(opt.modelName .. '.mean', mean)
torch.save(opt.modelName .. '.stdv', stdv)

if opt.minibatch then
    net = dsg_utils.TrainWithMinibatch(trainset, dsg_nets.VggBNDrop, 'msr', opt)
else
    net = dsg_utils.TrainNet(trainset, dsg_nets.Lenet, 'heuristic', opt.cuda)
end

torch.save(opt.modelName .. '.net', net)

-- create log file
cmd:log('log_main', opt)
