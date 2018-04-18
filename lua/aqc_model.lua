
function make_model_nin(opts)
    local pretrained_model=torch.load(opts.preload_model)
    local hu=opts.hu or 10
    local spatial_dropout=opts.spatial_dropout or 0
    local final_dropout=opts.final_dropout or 0 
    local feat=opts.feat or 3
    local ref=opts.ref
    local bottle=opts.bottle
    
    -- remove the last therteen layers
    pretrained_model:remove(#pretrained_model)
    pretrained_model:remove(#pretrained_model)
    pretrained_model:remove(#pretrained_model)
    pretrained_model:remove(#pretrained_model)
    pretrained_model:remove(#pretrained_model)
    pretrained_model:remove(#pretrained_model)
    pretrained_model:remove(#pretrained_model)
    pretrained_model:remove(#pretrained_model)
    pretrained_model:remove(#pretrained_model)
    pretrained_model:remove(#pretrained_model)
    pretrained_model:remove(#pretrained_model)
    pretrained_model:remove(#pretrained_model)
    pretrained_model:remove(#pretrained_model)
    -- the output data is now at 384x14x14 minibatch
    
    -- make it accept grayscale input 
    local old_first=pretrained_model:get(1)
    local new_first
    
    if ref then
      new_first=nn.SpatialConvolution(2, 96, 11, 11, 4,4, 5,5)
    
      new_first.weight[{{},{1},{},{}}]=old_first.weight[{{},1,{},{}}]+
                                      old_first.weight[{{},2,{},{}}]+
                                      old_first.weight[{{},3,{},{}}]
      new_first.weight[{{},{2},{},{}}]=new_first.weight[{{},{1},{},{}}]
      new_first.gradWeight[{{},{1},{},{}}]=old_first.gradWeight[{{},1,{},{}}]+
                                          old_first.gradWeight[{{},2,{},{}}]+
                                          old_first.gradWeight[{{},3,{},{}}]
      new_first.gradWeight[{{},{2},{},{}}]=new_first.gradWeight[{{},{1},{},{}}]
    else
      new_first=nn.SpatialConvolution(1, 96, 11, 11, 4,4, 5,5)
    
      new_first.weight[{{},{},{},{}}]=old_first.weight[{{},1,{},{}}]+
                                      old_first.weight[{{},2,{},{}}]+
                                      old_first.weight[{{},3,{},{}}]

      new_first.gradWeight[{{},{},{},{}}]=old_first.gradWeight[{{},1,{},{}}]+
                                          old_first.gradWeight[{{},2,{},{}}]+
                                          old_first.gradWeight[{{},3,{},{}}]
    end
    pretrained_model:remove(1)
    pretrained_model:insert(new_first,1)
    
    if ref then
      pretrained_model:insert(nn.View(-1,2,224,224),1) 
    else
      pretrained_model:insert(nn.View(-1,1,224,224),1) 
    end
    pretrained_model:insert(nn.Contiguous(),1)  -- input to bottle
    
    pretrained_model:add(nn.View(-1,384*feat,14,14)) -- bottle output FIX
    --
    local net=nn.Sequential()
    
    
    if not bottle then
      local prl=nn.Parallel(2,2)
      --
      for i=1,feat do
          prl:add(pretrained_model:clone())
      end
      --
      net:add(prl)
    else
      net:add(pretrained_model)
    end
        
    -- merge information from various views
    net:add(nn.SpatialConvolution(384*feat, 384, 1, 1))
    net:add(nn.SpatialBatchNormalization(384))
    net:add(nn.ReLU(true))
    
    -- replicating old steps from the model
    net:add(nn.SpatialMaxPooling(3,3, 2,2, 1,1))
    net:add(nn.SpatialConvolution(384, 1024, 3,3 , 1,1, 1,1))
    net:add(nn.SpatialBatchNormalization(1024))
    net:add(nn.ReLU(true))
    
    net:add(nn.SpatialConvolution(1024, hu, 1, 1))
    net:add(nn.SpatialBatchNormalization(hu))
    net:add(nn.ReLU(true))
    
    if spatial_dropout>0 then
      net:add(nn.SpatialDropout(spatial_dropout))
    end
    
    net:add(nn.SpatialConvolution(hu, hu, 7, 7, 1, 1))
    net:add(nn.SpatialBatchNormalization(hu))
    net:add(nn.ReLU(true))
    
    net:add(nn.SpatialConvolution(hu, hu, 1, 1, 1, 1))
    net:add(nn.SpatialBatchNormalization(hu))
    net:add(nn.ReLU(true))
    
    net:add(nn.View(-1,hu))
    if final_dropout>0 then
      net:add(nn.Dropout(final_dropout))
    end
    
    net:add(nn.Linear(hu,2))
    net:add(nn.LogSoftMax())

    cudnn.convert(net, cudnn)
--    cudnn.convert(model, cudnn)
    net=net:cuda()
    
    local criterion = nn.ClassNLLCriterion()
    criterion = criterion:cuda()
    
    return net,criterion
end

function make_model_r18(opts)
    local pretrained_model=torch.load(opts.preload_model)
    local hu=opts.hu or 10
    local spatial_dropout=opts.spatial_dropout or 0
    local final_dropout=opts.final_dropout or 0 
    local feat=opts.feat or 3
    local ref=opts.ref
    local bottle=opts.bottle
    
    -- remove the last three layers
    pretrained_model:remove(#pretrained_model)
    pretrained_model:remove(#pretrained_model)
    pretrained_model:remove(#pretrained_model)
    -- the output data is now at 512x7x7 minibatch
    
    -- make it accept grayscale input 
    local old_first=pretrained_model:get(1)
    
    local new_first

    if ref then
       new_first=cudnn.SpatialConvolution(2, 64, 7, 7, 2,2, 3,3)
       new_first:cuda()
       new_first.weight[{{},{1},{},{}}]=old_first.weight[{{},1,{},{}}]+
                                    old_first.weight[{{},2,{},{}}]+
                                    old_first.weight[{{},3,{},{}}]
       new_first.weight[{{},{2},{},{}}]=new_first.weight[{{},{1},{},{}}]
       
       new_first.gradWeight[{{},{1},{},{}}]=old_first.gradWeight[{{},1,{},{}}]+
                                        old_first.gradWeight[{{},2,{},{}}]+
                                        old_first.gradWeight[{{},3,{},{}}]
       new_first.gradWeight[{{},{2},{},{}}]=new_first.gradWeight[{{},{1},{},{}}]
    else
       new_first=cudnn.SpatialConvolution(1, 64, 7, 7, 2,2, 3,3)
       new_first:cuda()
       new_first.weight[{{},{},{},{}}]=old_first.weight[{{},1,{},{}}]+
                                    old_first.weight[{{},2,{},{}}]+
                                    old_first.weight[{{},3,{},{}}]

       new_first.gradWeight[{{},{},{},{}}]=old_first.gradWeight[{{},1,{},{}}]+
                                    old_first.gradWeight[{{},2,{},{}}]+
                                    old_first.gradWeight[{{},3,{},{}}]
    end

    pretrained_model:remove(1)
    pretrained_model:insert(new_first,1)
    
    --pretrained_model:insert(nn.Contiguous(),1)  -- input to bottle
    if ref then
      pretrained_model:insert(nn.View(-1,2,224,224),1)
    else
      pretrained_model:insert(nn.View(-1,1,224,224),1) 
    end
    
    pretrained_model:add(nn.View(-1,512*feat,7,7)) -- bottle output FIX
    --
    local net=nn.Sequential()
         
    if not bottle then
      local prl=nn.Parallel(2,2)
      --
      for i=1,feat do
          prl:add(pretrained_model:clone())
      end
      --
      net:add(prl)
    else
      net:add(pretrained_model)
    end
    
    -- merge information from different views
    net:add(nn.SpatialConvolution(512*feat, 512, 1, 1))
    net:add(nn.SpatialBatchNormalization(512))
    net:add(nn.ReLU(true))
    
    -- replicating old steps from the model
    net:add(nn.SpatialConvolution(512, hu, 1, 1))
    net:add(nn.SpatialBatchNormalization(hu))
    net:add(nn.ReLU(true))
    
    if spatial_dropout>0 then
      net:add(nn.SpatialDropout(spatial_dropout))
    end
    
    net:add(nn.SpatialConvolution(hu, hu, 7, 7, 1, 1))
    net:add(nn.SpatialBatchNormalization(hu))
    net:add(nn.ReLU(true))
    
    net:add(nn.SpatialConvolution(hu, hu, 1, 1, 1, 1))
    net:add(nn.SpatialBatchNormalization(hu))
    net:add(nn.ReLU(true))
    
    net:add(nn.View(-1,hu))
    if final_dropout>0 then
      net:add(nn.Dropout(final_dropout))
    end
    
    net:add(nn.Linear(hu,2))
    net:add(nn.LogSoftMax())

    cudnn.convert(net, cudnn)
    net=net:cuda()
    
    local criterion = nn.ClassNLLCriterion()
    criterion = criterion:cuda()
    
    return net,criterion
end

function make_model_simple(opts)
    local hu=opts.hu or 10
    local spatial_dropout=opts.spatial_dropout or 0
    local final_dropout=opts.final_dropout or 0 
    local feat=opts.feat or 3
    local bottle=opts.bottle
    --pretrained_model:insert(nn.Contiguous(),1)  -- input to bottle
    
    local single_model=nn.Sequential()
    
    single_model:insert(nn.View(-1,1,224,224),1) 
    
    single_model:add(nn.SpatialConvolution(1, 16, 5, 5 ))   -- 224-> 220
    single_model:add(nn.SpatialBatchNormalization(16))
    single_model:add(nn.ReLU(true))
    
    single_model:add(nn.SpatialConvolution(16, 16, 5, 5 ))  -- 220-> 216
    single_model:add(nn.SpatialBatchNormalization(16))
    single_model:add(nn.ReLU(true))
    
    single_model:add(nn.SpatialConvolution(16, 16, 5, 5 ))  -- 216-> 212
    single_model:add(nn.SpatialBatchNormalization(16))
    single_model:add(nn.ReLU(true))
    --
    single_model:add(nn.SpatialConvolution(16, 16, 5, 5, 3, 3 )) -- 212->70
    single_model:add(nn.SpatialBatchNormalization(16))
    single_model:add(nn.ReLU(true))

    single_model:add(nn.SpatialConvolution(16, 16, 5, 5, 5, 5 )) -- 70->14
    single_model:add(nn.SpatialBatchNormalization(16))
    single_model:add(nn.ReLU(true))
    
    local net=nn.Sequential()
    --net:add(nn.SplitTable(1,3))
    
    --local bottle=nn.Bottle(pretrained_model,3,3) --forward all data through the same net
    -- 
    --prl:add(pretrained_model:clone())
    --prl:add(pretrained_model:clone())
    --prl:add(pretrained_model:clone())
    --
    --net:add(bottle)
    if bottle then
        single_model:add(nn.View(-1,feat*16,14,14))
        net:add(single_model)
    else
        local prl=nn.Parallel(2,2)
        local i
        for i=1,feat do
            prl:add(single_model:clone())
        end
        net:add(prl)
    end
    
    -- merge information from various modalities
    net:add(nn.SpatialConvolution(16*feat, 128, 1, 1))
    net:add(nn.SpatialBatchNormalization(128))
    net:add(nn.ReLU(true))
    
    -- replicating old steps from the model
    net:add(nn.SpatialConvolution(128, hu, 1, 1))
    net:add(nn.SpatialBatchNormalization(hu))
    net:add(nn.ReLU(true))
    
    if spatial_dropout>0 then
      net:add(nn.SpatialDropout(spatial_dropout))
    end
    
    net:add(nn.SpatialConvolution(hu, hu, 5, 5, 3, 3)) -- 14->4
    net:add(nn.SpatialBatchNormalization(hu))
    net:add(nn.ReLU(true))
    
    net:add(nn.SpatialConvolution(hu, hu, 4, 4, 1, 1)) -- 4->1
    net:add(nn.SpatialBatchNormalization(hu))
    net:add(nn.ReLU(true))
    
    net:add(nn.SpatialConvolution(hu, hu, 1, 1, 1, 1))
    net:add(nn.SpatialBatchNormalization(hu))
    net:add(nn.ReLU(true))
    
    net:add(nn.View(-1,hu))
    if final_dropout>0 then
      net:add(nn.Dropout(final_dropout))
    end
    
    net:add(nn.Linear(hu,2))
    net:add(nn.LogSoftMax())

    cudnn.convert(net, cudnn)
--    cudnn.convert(model, cudnn)
    net=net:cuda()
    
    local criterion = nn.ClassNLLCriterion()
    criterion = criterion:cuda()
    
    return net,criterion
end
