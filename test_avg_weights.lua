require 'torch'

dsg_utils = require 'dsg_utils'

cmd = torch.CmdLine()
cmd:addTime()
cmd:option('-submissionName', 'submission', 'name of the submission file')
cmd:option('-cuda', false, 'if true train with minibatches')

opt = cmd:parse(arg or {})

local models = {'model1', 'model2'}
local nmodels = #models

-- Load models
nets = {}

for k,v in ipairs(models) do
    net = torch.load(v .. '.net')
    net:evaluate()
    table.insert(nets, net)
end

mean = torch.load(models[1] .. '.mean')
stdv = torch.load(models[1] .. '.stdv')

-- Load test set
--dsg_utils.PreprocessAndAugmentDataset("sample_submission4.csv", "dsg_test.t7", "rgb")
testset = torch.load("dsg_test.t7")
local ntest = testset.label:size(1)

if opt.cuda then
    testset.data = testset.data:cuda()
    testset.label = testset.label:cuda()
end

-- Average weights of nets
print("Average weights")
params, gradParamas = nets[1]:getParameters()
for i=2,#nets do
    params2, gradParams2 = nets[i]:getParameters()
    params:add(params2)
end

params:div(#nets)

print("Testing")

for i = 1,3 do
    testset.data[{ {}, {i}, {}, {} }]:add(-mean[i])
    testset.data[{ {}, {i}, {}, {} }]:div(stdv[i])
end

local file = assert(io.open(opt.submissionName .. '.csv', "w"))
local file_detailed = assert(io.open(opt.submissionName .. '_detailed.csv', "w"))
file:write("Id,label\n")
file_detailed:write("Id,label,cat1,cat2,cat3,cat4\n")

for i=1,ntest do
    local prediction = net:forward(testset.data[i])
    prediction:exp()

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
