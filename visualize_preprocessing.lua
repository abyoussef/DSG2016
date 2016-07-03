require 'csvigo'
require 'image'
preprocessing = require 'preprocessing_nagadomi'

local size = 32
local params = torch.load('dsg_params.t7')
local parsed = csvigo.load({path='id_train.csv', mode="query"})
local dataset = parsed('all')

local data = torch.Tensor(1, 3, size, size)
local n = 20
local before = torch.Tensor(3, n * (size + 5), n * (size + 5)):zero()
local after = torch.Tensor(3, n * (size + 5), n * (size + 5)):zero()

for i = 1,n * n do
    local img = image.scale(image.load('roof_images/' .. dataset.Id[i] .. '.jpg'), size, size)

    local row = math.ceil(i / n)
    local col = i - (row - 1) * n

    local img0 = before:narrow(2, 1 + (row - 1) * (size + 5), size):narrow(3, 1 + (col - 1) * (size + 5), size)
    img0:copy(img)

    data[1] = img
    preprocessing.preprocessing(data, params)
    local img1 = after:narrow(2, 1 + (row - 1) * (size + 5), size):narrow(3, 1 + (col - 1) * (size + 5), size)
    img1:copy(data[1])
end

image.display({image=before, legend='before'})
image.display({image=after, legend='after'})
