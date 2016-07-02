require 'csvigo'
require 'image'
require 'xlua'

local parsed = csvigo.load({path='id_train.csv', mode="query"})
local dataset = parsed('all')
local ndata = #dataset.Id
local cont = {0, 0, 0, 0}

for k,v in ipairs(dataset.Id) do
    xlua.progress(k, ndata)
    img = image.load('roof_images/' .. v .. '.jpg')
    local label = tonumber(dataset.label[k])
    image.save('class' .. label .. '/' .. v .. '.jpg', img)
    cont[label] = cont[label] + 1
end

print(cont)
