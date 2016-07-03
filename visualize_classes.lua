require 'csvigo'
require 'image'

local size = 32
local parsed = csvigo.load({path='id_train.csv', mode="query"})
local dataset = parsed('all')
local ndata = #dataset.Id
local nh, nw = 20, 20

local function visualize_class(label)
    local res = torch.Tensor(3, nh * (size + 5), nw * (size + 5)):zero()
    local cont = 0

    for i = 1,ndata do
        if dataset.label[i] == label then
            cont = cont + 1
            local img = image.scale(image.load('roof_images/' .. dataset.Id[i] .. '.jpg'), size, size)

            local row = math.ceil(cont / nw)
            local col = cont - (row - 1) * nw

            local aux = res:narrow(2, 1 + (row - 1) * (size + 5), size):narrow(3, 1 + (col - 1) * (size + 5), size)
            aux:copy(img)
        end

        if cont == nh * nw then
            break
        end
    end

    image.display({image=res, legend=label})
end

visualize_class('1')
visualize_class('2')
visualize_class('3')
visualize_class('4')
