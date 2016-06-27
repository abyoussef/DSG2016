require 'torch'

dsg_utils = require 'dsg_utils'

local cuda_flag = false

model_files = {'model1.net', 'model2.net'}
nmodels = #model_files

-- Load models
nets = {}

for k,v in ipairs(model_files) do
    net = torch.load(v)
    if cuda_flag then
        net:cuda()
    end
    table.insert(nets, net)
end

mean = torch.load('model.mean')
stdv = torch.load('model.stdv')

-- Load test set
testset = dsg_utils.LoadDataset("sample_submission4.csv")
local ntest = #testset.Id

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

local filename = 'submission.csv'
local file = assert(io.open(filename, "w"))
file:write("Id,label\n")

for i=1,ntest do
    local prediction = 0
    for k,v in ipairs(nets) do
        prediction = prediction + net:forward(testset.data[i])
    end
    prediction = prediction / nmodels

    local confidences, indices = torch.sort(prediction, true) -- sort in descending order

    file:write(testset.Id[i] .. "," .. indices[1] .. "\n")
end

file:close()
print("Testing finished")
