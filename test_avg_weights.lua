require 'nn'
require 'cunn'

cmd = torch.CmdLine()
cmd:addTime()
cmd:option('-submissionName', 'submission', 'name of the submission file')
cmd:option('-loadPath', 'results/', 'path to the models')
cmd:option('-savePath', 'results/', 'path to the save directory')
cmd:option('-cuda', false, 'if true train with minibatches')

opt = cmd:parse(arg or {})

-- create log file
cmd:log(opt.savePath .. 'test_avg_weights_' .. opt.submissionName .. '.log', opt)

-- Load models
local models = {'model1', 'model2'}
local nmodels = #models
nets = {}

for k,v in ipairs(models) do
    net = torch.load(opt.loadPath .. v .. '.net')
    net:evaluate()
    if opt.cuda then
        net = net:cuda()
        testset.data = testset.data:cuda()
    end
    table.insert(nets, net)
end

-- Load test set
testset = torch.load("dsg_test.t7")
local ntest = testset.label:size(1)

if opt.cuda then
    testset.data = testset.data:cuda()
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

local file = assert(io.open(opt.savePath .. opt.submissionName .. '.csv', "w"))
local file_detailed = assert(io.open(opt.savePath .. opt.submissionName .. '_detailed.csv', "w"))
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
