require 'io'
require 'torch'
require 'cutorch'
require 'paths'
require 'sys'
require 'xlua'

local sqlite3 = require('lsqlite3complete')

local image = require 'image'
local math  = require 'math'

-- TODO: configure DB location
local qc_db=nil -- sqlite3.open('data/qc_db.sqlite3')

function load_qc_db(data_prefix, feat)
    -- load training list
    local feat=feat or 3
    local samples={}
    local status=2
    local subjects={}
    
    qc_db=qc_db or sqlite3.open(data_prefix..'/qc_db.sqlite3')
    -- populate table with locations of QC jpg files
    for line in qc_db:rows("select variant,cohort,subject,visit,path,xfm,pass from qc_all " ) do
      --string.gsub(line, "\n", "")
        local variant,cohort,subject,visit,path,xfm,pass=unpack(line)
        
        if pass=='TRUE' then status=1 else status=0 end
        
        local id=string.format('%s_%s_%s_%s',variant,cohort,subject,visit)
        local qc={}
        local i
        
        for i=0,(feat-1) do
            qc[#qc+1]=string.format('%s/%s/qc/aqc_%s_%s_%d.jpg', data_prefix, path, subject, visit, i)
        end
        samples[#samples + 1] = { id, status, qc, variant, cohort, subject, visit }
    end
    
    -- make a list of all subjects
    for line in qc_db:rows("select distinct subject from qc_all" ) do
        subjects[#subjects+1]=line[1]
    end
    return samples,subjects
end

function load_qc_data(samples)
    local dataset={}
    local _samples={}
    for i,j in ipairs(samples) do
        local id,status,qc=unpack(j)
        local f
        if paths.filep( qc[1] ) and paths.filep( qc[2] ) and paths.filep( qc[3] ) then
            _samples[#_samples+1]=j
        else
          print(string.format("Check: %s %s %s ",qc[1],qc[2],qc[3]))
        end
    end
    return _samples
end

function allocate_minibatch(ds, samples, ref ) 
  local feat=3
  if ref ~= nil then
      feat=6
  end
  
  local out1=torch.CudaTensor(     samples, feat, 224, 224 )
  local out2=torch.CudaByteTensor( samples )
  
  return {out1,out2}
end



function get_minibatch_samples(minibatch, dataset1, dataset2, samples, state, mean_sd, ref)

  if  state.shuffle1 == nil then
      state.length1=#dataset1
      state.shuffle1=torch.randperm(#dataset1):long()
      state.idx1=1
      
      state.length2=#dataset2
      state.shuffle2=torch.randperm(#dataset2):long()
      state.idx2=1
  end
  
  local stride=1
  if ref ~= nil then
      stride=2
  end
  
  local s
  
  for s=0,(samples/2-1) do
      local p1 = state.shuffle1[state.idx1]
      state.idx1 = state.idx1 % state.length1 + 1
      
      local p2 = state.shuffle2[state.idx2]
      state.idx2 = state.idx2 % state.length2 + 1
      local f
      

      for f=0,2 do
        
        local img=image.load( dataset1[ p1 ][3][f+1], 1, 'float' )
        img:add(-mean_sd.mean)
        img:mul(1/mean_sd.sd)
        minibatch[1][ { s*2+1, stride*f+1, {},{} } ]:copy(img:view(224, 224))
        
        
        local img=image.load( dataset2[ p2 ][3][f+1], 1, 'float' )	
        img:add(-mean_sd.mean)
        img:mul(1/mean_sd.sd)
        minibatch[1][ { s*2+2, stride*f+1, {},{} } ]:copy(img:view(224, 224))
        
        if ref~=nil then
             minibatch[1][ { s*2+1, stride*f+2, {},{} } ]:copy(ref[f+1]:view(224, 224) )
             minibatch[1][ { s*2+2, stride*f+2, {},{} } ]:copy(ref[f+1]:view(224, 224) )
        end
      end
      
      minibatch[2][ s*2+1 ]=dataset1[ p1 ][2]+1
      minibatch[2][ s*2+2 ]=dataset2[ p2 ][2]+1
  end
end

function get_sample(minibatch, dataset, mean_sd, qc_ref)
  local f
  local stride=1
  
  if qc_ref ~= nil then
      stride=2
  end
  
  for f=0,2 do
    local img=image.load(dataset[3][f+1], 1, 'float')
    
    img:add(-mean_sd.mean)
    img:mul(1/mean_sd.sd)
    
    minibatch[1][ { 1,f*stride+1,{},{} } ]:copy(         img:view(224, 224) )
    
    if qc_ref ~=nil then
        minibatch[1][ { 1,f*stride+2,{},{} } ]:copy( qc_ref[f+1]:view(224, 224) )
    end
  end
  minibatch[2][ 1 ]=dataset[2]+1  
end

function subrange(t, first, last)
  local sub = {}
  local i
  for i=first,last do
    sub[#sub + 1] = t[i]
  end
  return sub
end


function calculate_mean_sd(dataset, first)
    local first=first or #dataset
    local _mean=0.0
    local _sd=0.0
    local j,f
    
    for j=1,first do
        for f=1,3 do
            local img=image.load(dataset[ j ][3][f], 1, 'float')
            _mean=_mean+torch.mean(img)
            _sd=_sd+torch.std(img)
        end
    end
    
    _mean=_mean/(first*3)
    _sd=_sd/(first*3)
    
    return {mean=_mean,sd=_sd}
end


function save_progress(prog,out)
    local f = assert(io.open(out, 'w'))
    f:write("batch,accuracy,error,tpr,fpr,v_accuracy,v_error,v_tpr,v_fpr\n")
    local _,i
    for _,i in ipairs(prog) do
        f:write(string.format("%d,%f,%f,%f,%f,%f,%f,%f,%f\n", i.batch or 0,i.acc or 0,i.err or 0,i.fpr or 0,i.tpr or 0,i.v_acc or 0,i.v_err or 0,i.v_tpr or 0,i.v_fpr or 0))
    end
    f:close()
end

function accuracy_max(result,ground_truth)
    local _,_result=result:max(2) -- last dimension contains the outputs
    return torch.sum(torch.eq(_result:byte(),ground_truth:byte()))/ground_truth:nElement()
end

function accuracy_tpr_fpr_max(result,ground_truth)
    local _,_result=result:max(2) -- last dimension contains the outputs
    
    -- make them zero-based
    local _ground_truth=ground_truth:clone():byte()
    _result=_result:byte()
    
    local tp=torch.dot(_ground_truth:eq(2),_result:eq(2))
    local tn=torch.dot(_ground_truth:eq(1),_result:eq(1))
    local fn=torch.dot(_ground_truth:eq(2),_result:eq(1))
    local fp=torch.dot(_ground_truth:eq(1),_result:eq(2))
    
    local acc=(tp+tn)/ground_truth:nElement()
    local tpr=(tp)/(tp+fn)
    local fpr=(fp)/(tn+fp)
    
    return acc,tpr,fpr
end

function load_ref_data( img )
    local i,j
    local ret={}
    
    for i,j in ipairs(img) do
        ret[i]=image.load( j, 1, 'float' )
    end
    
    return ret
end
