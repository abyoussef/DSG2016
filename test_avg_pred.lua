require 'torch'

dsg_utils = require 'dsg_utils'

local cuda_flag = false
local models = {'model1', 'model2'}
local submission_name = 'submission'

nmodels = #models

-- Load models
nets = {}

for k,v in ipairs(models) do
    net = torch.load(v .. '.net')
    if cuda_flag then
        net:cuda()
    end
    table.insert(nets, net)
end

mean = torch.load(models[1] .. '.mean')
stdv = torch.load(models[1] .. '.stdv')

-- Load test set
--dsg_utils.PreprocessAndAugmentDataset("sample_submission4.csv", "dsg_test.t7", "rgb")
testset = torch.load("dsg_test.t7")
local ntest = #testset.label

-- Using CUDA
if cuda_flag then
    require 'cunn'
    testset.data = testset.data:cuda()
    testset.label = testset.label:cuda()
end

print("Testing")

for i = 1,3 do
    testset.data[{ {}, {i}, {}, {} }]:add(-mean[i])
    testset.data[{ {}, {i}, {}, {} }]:div(stdv[i])
end

local file = assert(io.open(submission_name .. '.csv', "w"))
local file_detailed = assert(io.open(submission_name .. '_detailed.csv', "w"))
file:write("Id,label\n")
file_detailed:write("Id,label,cat1,cat2,cat3,cat4\n")

for i=1,ntest do
    local prediction = torch.Tensor(4):zero()
    for k,v in ipairs(nets) do
        local net_prediction = net:forward(testset.data[i])
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
