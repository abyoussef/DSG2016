require 'cunn'
dsg_utils = require 'dsg_utils'

cmd = torch.CmdLine()
cmd:addTime()
cmd:option('-modelName', 'model', 'name of the model')
cmd:option('-submissionName', 'submission', 'name of the submission file')
cmd:option('-batchSize', 100, 'size of batches')
cmd:option('-epoch', 0, 'epoch that will be tested')
cmd:option('-cuda', false, 'if true train with minibatches')

opt = cmd:parse(arg or {})

-- Load model
if opt.epoch == 0 then
    net_name = opt.modelName
else
    net_name = opt.modelName .. '_epoch_' .. opt.epoch
end
net = torch.load(net_name .. '.net')
net:evaluate()

-- Load test set
testset = torch.load("dsg_test.t7")
local ntest = testset.data:size(1)

local file = assert(io.open(opt.submissionName .. '.csv', "w"))
local file_detailed = assert(io.open(opt.submissionName .. '_detailed.csv', "w"))
file:write("Id,label\n")
file_detailed:write("Id,label,cat1,cat2,cat3,cat4\n")

for i=1,ntest,opt.batchSize do
    xlua.progress(i, ntest)

    local bs = math.min(opt.batchSize, ntest + 1 - i)
    local inputs = torch.FloatTensor(bs, 3, 32, 32)
    inputs:copy(testset.data:narrow(1, i, bs))
    if opt.cuda then
        inputs = inputs:cuda()
    end

    local prediction = net:forward(inputs)
    assert(prediction:size(1) == bs and prediction:size(2) == 4 and prediction:dim() == 2)
    prediction:exp()

    for j=1,bs do
        local confidences, indices = torch.sort(prediction[j], true) -- sort in descending order

        file:write(testset.Id[i + j - 1] .. "," .. indices[1] .. "\n")
        file_detailed:write(testset.Id[i + j - 1] .. "," .. indices[1])
        for k = 1,4 do
            file_detailed:write("," .. prediction[j][k])
        end
        file_detailed:write("\n")
    end
end
xlua.progress(ntest, ntest)

file:close()

-- create log file
cmd:log('log_test_net', opt)
