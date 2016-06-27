require 'torch'
require 'image'
require 'nn'
dsg_utils = require 'dsg_utils'
dsg_nets = require 'dsg_nets'

local cuda_flag = false
local test = false

if cuda_flag then
    require 'cunn'
end

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
mean, std = dsg_utils.Normalize(trainset)
net = dsg_utils.TrainNet(trainset, dsg_nets.Lenet, cuda_flag)

-- Test the network

if test then
	-- Load test set

	testset = dsg_utils.LoadDataset("sample_submission4.csv")
	local ntest = #testset.Id

	print("Testing")

	for i = 1,3 do
	    testset.data[{ {}, {i}, {}, {} }]:add(-mean[i])
	    testset.data[{ {}, {i}, {}, {} }]:div(stdv[i])
	end

	--classes = {"North-South", "East-West", "Flat roof" , "Other"}

	--rtest = math.random(ntest)
	--predicted = net:forward(testset.data[rtest])
	--predicted:exp() -- convert log-probability to probability
	--for i = 1,predicted:size(1) do
	--    print(classes[i], predicted[i])
	--end
	--image.display(testset.data[rtest])

	local filename = 'submission.csv'
	local file = assert(io.open(filename, "w"))
	file:write("Id,label\n")

	for i=1,ntest do
	    local prediction = net:forward(testset.data[i])
	    local confidences, indices = torch.sort(prediction, true) -- sort in descending order

	    file:write(testset.Id[i] .. "," .. indices[1] .. "\n")
	end

	file:close()
	print("Testing finished")
end