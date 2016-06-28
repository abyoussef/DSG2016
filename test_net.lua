dsg_utils = require 'dsg_utils'

local cuda_flag = false
local model_name = 'model'
local submission_name = 'submission'

-- Load model
net = torch.load(model_name .. '.net')
mean = torch.load(model_name .. '.mean')
stdv = torch.load(model_name .. '.stdv')
net:evaluate()

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

--classes = {"North-South", "East-West", "Flat roof" , "Other"}

--rtest = math.random(ntest)
--predicted = net:forward(testset.data[rtest])
--predicted:exp() -- convert log-probability to probability
--for i = 1,predicted:size(1) do
--    print(classes[i], predicted[i])
--end
--image.display(testset.data[rtest])

local file = assert(io.open(submission_name .. '.csv', "w"))
local file_detailed = assert(io.open(submission_name .. '_detailed.csv', "w"))
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
