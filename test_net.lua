require 'cunn'
dsg_utils = require 'dsg_utils'

cmd = torch.CmdLine()
cmd:addTime()
cmd:option('-modelName', 'model', 'name of the model')
cmd:option('-submissionName', 'submission', 'name of the submission file')
cmd:option('-cuda', false, 'if true train with minibatches')
cmd:option('-float', false, 'if true cas to float')

opt = cmd:parse(arg or {})

local submission_name = 'submission'

-- Load model
net = torch.load(opt.modelName .. '.net')
net:evaluate()

-- Load test set
testset = torch.load("dsg_test.t7")
local ntest = testset.data:size(1)

if opt.float then
    testset.data = testset.data:float()
end

-- Using CUDA
if opt.cuda then
    testset.data = testset.data:cuda()
end

print("Testing")

--classes = {"North-South", "East-West", "Flat roof" , "Other"}

--rtest = math.random(ntest)
--predicted = net:forward(testset.data[rtest])
--predicted:exp() -- convert log-probability to probability
--for i = 1,predicted:size(1) do
--    print(classes[i], predicted[i])
--end
--image.display(testset.data[rtest])

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

-- create log file
cmd:log('log_test_net', opt)
