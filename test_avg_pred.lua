require 'nn'
require 'cunn'

cmd = torch.CmdLine()
cmd:addTime()
cmd:option('-submissionName', 'submission', 'name of the submission file')
cmd:option('-cuda', false, 'if true train with minibatches')

opt = cmd:parse(arg or {})

-- create log file
cmd:log('log_test_avg_pred_' .. opt.submissionName .. '.log', opt)

-- Load models
local models = {'model1', 'model2'}
local nmodels = #models
nets = {}

for k,v in ipairs(models) do
    net = torch.load(v .. '.net')
    net:evaluate()
    if opt.cuda then
        net = net:cuda()
    end
    table.insert(nets, net)
end

mean = torch.load(models[1] .. '.mean')
stdv = torch.load(models[1] .. '.stdv')

-- Load test set
testset = torch.load("dsg_test.t7")
local ntest = testset.label:size(1)

-- Using CUDA
if opt.cuda then
    testset.data = testset.data:cuda()
end

print("Testing")

local file = assert(io.open(opt.submissionName .. '.csv', "w"))
local file_detailed = assert(io.open(opt.submissionName .. '_detailed.csv', "w"))
file:write("Id,label\n")
file_detailed:write("Id,label,cat1,cat2,cat3,cat4\n")

for i=1,ntest do
    local prediction = torch.Tensor(4):zero()
    if opt.cuda then
        prediction = prediction:cuda()
    end
    for k,v in ipairs(nets) do
        local net_prediction = nets[k]:forward(testset.data[i])
        net_prediction:exp()
        prediction:add(net_prediction)
    end
    prediction:div(nmodels)

    local confidences, indices = torch.sort(prediction, true) -- sort in descending order

    file:write(testset.Id[i] .. "," .. indices[1] .. "\n")
    file_detailed:write(testset.Id[i] .. "," .. indices[1])
    for i = 1,4 do
        file_detailed:write("," .. prediction[i])
    end
    file_detailed:write("\n")
end

file:close()
print("Testing finished")
