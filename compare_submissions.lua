require 'csvigo'
require 'image'

local submission_name1 = 'submission1'
local submission_name2 = 'submission2'
local limit = 10

local sub1_csv = csvigo.load({path=submission_name1 .. '.csv', mode="query"})
local sub1 = sub1_csv('all')
local sub2_csv = csvigo.load({path=submission_name2 .. '.csv', mode="query"})
local sub2 = sub2_csv('all')

for k,v in ipairs(sub1.Id) do
    if sub1.label[k] ~= sub2.label[k] and limit > 0 then
        local id = sub1.Id[k]
        local i = image.load('roof_images/' .. id .. '.jpg')
        local legend = 'sub1 = ' .. sub1.label[k] .. ', sub2 = ' .. sub2.label[k]
        print(sub1['cat1'] .. ' ' .. sub1['cat2'] .. ' ' .. sub1['cat3'] .. ' ' .. sub1['cat4'])
        print(sub2['cat1'] .. ' ' .. sub2['cat2'] .. ' ' .. sub2['cat3'] .. ' ' .. sub2['cat4'])
        image.display{image=i, legend=legend}
        limit = limit - 1
    end
end
