dsg_utils = require 'dsg_utils'

local cuda_flag = false

-- Load model
net = torch.load('model.net')
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
