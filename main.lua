require 'torch'
require 'image'
require 'nn'
dsg_utils = require 'dsg_utils'
dsg_nets = require 'dsg_nets'

cmd = torch.CmdLine()
cmd:addTime()
cmd:option('-modelName', 'model', 'name of the model')
cmd:option('-augmentedData', false, 'if true use augmented data')
cmd:option('-init', 'none', 'if true use augmented data')
cmd:option('-minibatch', false, 'if true train with minibatches')
cmd:option('-batchSize', 64, 'size of batches')
cmd:option('-nEpochs', 100, 'number of epochs')
cmd:option('-epochLearningStep', 1, 'number of epochs between learning rate change')
cmd:option('-epochSaveStep', 50, 'number of epochs between model save')
cmd:option('-resultsSavePath', 'results/', 'path to the save directory for result files')
cmd:option('-modelsSavePath', 'models/', 'path to the save directory for models')
cmd:option('-cuda', false, 'if true cast to cuda')
cmd:option('-cudnn', false, 'if true use cudnn')
cmd:option('-float', false, 'if true cast to float')

opt = cmd:parse(arg or {})

-- create log file
cmd:log(opt.resultsSavePath .. 'main_' .. opt.modelName .. '.log', opt)

-- Load training set
if opt.augmentedData then
    trainset = torch.load("dsg_train_augmented.t7")
else
    trainset = torch.load("dsg_train.t7")
end

--net, mean, std = dsg_utils.KFoldedCV(trainset, 4, dsg_nets.Lenet, opt.cuda)

if opt.minibatch then
	testset = torch.load("dsg_test.t7")
    net = dsg_utils.TrainWithMinibatch(trainset, testset, dsg_nets.VggBNDrop, opt)
else
    net = dsg_utils.TrainOnline(trainset, dsg_nets.VggBNDrop, opt)
end

torch.save(opt.modelsSavePath .. opt.modelName .. '.net', net)
