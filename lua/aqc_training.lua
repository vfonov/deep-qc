require 'nn'
require 'cunn'
require 'cudnn'
require 'paths'
require 'optim'
display = require 'display'
torch.setdefaulttensortype('torch.FloatTensor')

--
paths.dofile('aqc_data.lua')
paths.dofile('aqc_model.lua')


function parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-based Auto QC testing script ')
   cmd:text()
   cmd:text('Options:')
    ------------ General options --------------------
   cmd:option('-ref',  false, 'Use reference images')
   cmd:option('-r18',  false, 'Use resnet-18, otherwise NiN')
   cmd:option('-reset',false, 'Reset pretrained weights (start from scratch)')
   cmd:option('-prefix',  '', 'Output prefix')
   cmd:option('-run',  1, 'Run folds')
   cmd:option('-batches',  1500, 'Number of batches')
   cmd:option('-data',  'data', 'Location of input data')
   cmd:option('-minibatch', 64, 'Minibatch size')
   cmd:text()
   
   local opt = cmd:parse(arg or {})
   return opt
end

local opt=parse(arg)
if opt.prefix=="" then
    print(string.format("Specify -prefix"))
    return 1
end


torch.manualSeed(4)

local model_variant = opt.prefix
local input_prefix  = opt.data

-- load QC results tables
local qc_pp,qc_subjects = load_qc_db(input_prefix)

-- preshuffle all samples 
do
    local samples_shuffle=torch.randperm(#qc_pp):long()
    local i
    local _qc_pp={}
    for i=1,#qc_pp do
        _qc_pp[#_qc_pp+1]=qc_pp[ samples_shuffle[ i ] ]
    end
    qc_pp=_qc_pp
end

-- preshuffle all subjects
do
    local subjects_shuffle=torch.randperm(#qc_subjects):long()
    local i
    local _qc_subjects={}
    for i=1,#qc_subjects do
        _qc_subjects[#_qc_subjects+1]=qc_subjects[ subjects_shuffle[ i ] ]
    end
    qc_subjects=_qc_subjects
end

local qc_ref=load_ref_data({input_prefix.."/mni_icbm152_t1_tal_nlin_sym_09c_0.jpg",
                            input_prefix.."/mni_icbm152_t1_tal_nlin_sym_09c_1.jpg",
                            input_prefix.."/mni_icbm152_t1_tal_nlin_sym_09c_2.jpg"})

local use_samples=opt.minibatch
local use_iter=2

local program={
    { batches=opt.batches, iter=use_iter, LR=0.0001 ,LRd=0.0001, beta1=0.9, beta2=0.999, eps=1e-8, samples=use_samples, validate=10, optim=optim.adam, rehash=500},
}

local model_opts = {
    spatial_dropout=0.0,
    final_dropout=0.5,
    hu=32,
    feat=3,
    bottle=true,
    ref=opt.ref
}

if opt.r18 then
    model_opts.preload_model=input_prefix..'/models/resnet-18.t7'
    print("Using resnet-18")
else
    model_opts.preload_model=input_prefix..'/models/nin_.t7'
    print("Using NiN")
end

local momentum=0.9  -- momentum
local WD=0.0001       -- weight decay
local nesterov=false

local validation=200
local folds=8
local run_folds=opt.run

local fold=0
local all_samples=qc_pp
local dataset=qc_pp
local display_sample={}

print(string.format("Total samples:%d",#dataset))
print(string.format("Total number of subjects:%d",#qc_subjects))

-- normalization is outside of CV loop now
-- TODO: use random subset here
local mean_sd=calculate_mean_sd(dataset,2000)
print(mean_sd.mean, mean_sd.sd)
torch.save(model_variant.."_mean_sd.t7", mean_sd)

do
    local i
    for i=1,3 do
        qc_ref[i]:add( -mean_sd.mean  )
        qc_ref[i]:mul( 1.0/mean_sd.sd )
    end
end

if not opt.ref then
    qc_ref=nil
end

-- CV starts here
-- TODO: use subjects for CV
for fold=0,(run_folds-1) do
  local i
  
  local fold_prefix=string.format("%s_%02d_",model_variant,fold+1)

  local testing_range_lo=fold*math.floor(#qc_subjects/folds)+1
  local testing_range_hi=(fold+1)*math.floor(#qc_subjects/folds)
  
  local test_subjects={}
  
  -- make index of test data
  for i=testing_range_lo,testing_range_hi do -- add leftovers
    test_subjects[qc_subjects[i]]=1
  end
  
  local test_dataset={}

  -- split data
  local train_dataset={}

  for i=1,#dataset do
    if test_subjects[ dataset[i][6] ] then -- 
        test_dataset[#test_dataset+1]=dataset[i]
    else
        train_dataset[#train_dataset+1]=dataset[i]
    end
  end

  local display_error = {
      title = string.format("Training Error, fold %d/%d",fold+1,run_folds),
      labels = {"batch", "training","validation"},
      ylabel =  "-log10(loss)",
  }
  local error_data={}

  local display_accuracy = {
      title = string.format("Training Accuracy, fold %d/%d",fold+1,run_folds),
      labels = {"batch", "training_acc","val_acc","val_tpr","val_fpr"},
      ylabel = "Rate",
  }
  local accuracy_data={}

  local validation_dataset=subrange(train_dataset,1,validation)
  train_dataset     = subrange(train_dataset,validation+1,#train_dataset)
  
  -- split train dataset into positive and negative
  
  local train_dataset_pos={}
  local train_dataset_neg={}
  
  for i=1,#train_dataset do
      if train_dataset[i][2]>0 then
        train_dataset_pos[#train_dataset_pos+1]=train_dataset[ i ]
      else
        train_dataset_neg[#train_dataset_neg+1]=train_dataset[ i ]
      end
  end

  local model,criterion
  
  if opt.r18 then
     model,criterion=make_model_r18( model_opts )
  else
      model,criterion=make_model_nin( model_opts )
  end

  -- WARNING: Disabling!
  if opt.reset then
    for i,module in ipairs(model:listModules()) do
       module:reset()
    end
  end

  print(model)
  print("")
  print(string.format("Train dataset:%d Validation dataset:%d testing dataset:%d pos:%d neg:%d",#train_dataset, #validation_dataset, #test_dataset,#train_dataset_pos,#train_dataset_neg))
  print("")
  do -- TODO move this into function ?
      local total_batches=0
      local _,p
      for _,p in ipairs(program) do
          total_batches=total_batches+p.batches
      end
      print(string.format("Fold:%d Running optimization using %d batches, %d each",fold+1,total_batches,use_samples))
    
      local final_model_name=fold_prefix..'training.t7'
    
      local iter_progress={}
      model:training() -- enable dropout
      local batch=0
      local _,p
      --local timer = torch.Timer()
      local parameters, gradParameters = model:getParameters()
    
      local sampler_state={} -- state for sampler
    
      local op_config=nil
      local op_state={}
    
      local val_error=0.0
      local val_acc=0.0
      local val_tpr=0.0
      local val_fpr=0.0
      local op=optim.adam
    
      for _,p in ipairs(program) do
        
          local minibatch=allocate_minibatch(train_dataset, p.samples or 1, qc_ref) -- allocate data in GPU
        
          if op_config==nil or p.reset then
              op_config = {
                  learningRate = p.LR,
                  learningRateDecay = p.LRd,
                  momentum    = p.momentum ,
                  dampening   = p.dampening,
                  weightDecay = p.WD or WD,
                  beta1       = p.beta1,
                  beta2       = p.beta2,
                  epsilon     = p.eps,
                  nesterov    = p.nesterov or nesterov
              }
              op_state = {  }
          end
        
        
          if p.optim then op=p.optim end
        
          local j
          --xlua.progress(batch,total_batches)
          for j = 1,p.batches do
              collectgarbage()
              
              if p.rehash and j%p.rehash==1 then
                  sampler_state={}
              end
              
              batch=batch+1
              get_minibatch_samples(minibatch, train_dataset_pos, train_dataset_neg, p.samples, sampler_state, mean_sd, qc_ref)
              local avg_err=0
              local last_err=0
              local i
              for i=1,p.iter do
                  local outputs
                  local err
                  local acc,tpr,fpr
                
                  feval = function(x)
                      model:zeroGradParameters()
                    
                      outputs = model:forward(minibatch[1])
                      err = criterion:forward(outputs, minibatch[2])
                      local gradOutputs = criterion:backward(outputs, minibatch[2])
                      model:backward( minibatch[1], gradOutputs)
                    
                      return err, gradParameters
                  end
                  op(feval,parameters,op_config,op_state)
                  avg_err=avg_err+err
                  last_err=err
              end

              avg_err=-math.log10(avg_err/p.iter)
              
              local outputs=model:forward(minibatch[1])
              local acc,tpr,fpr=accuracy_tpr_fpr_max(outputs, minibatch[2])
              
              if j%p.validate==0 and validation>0 then -- time to run validation experiment
                  
                  display_sample.title=string.format("Batch %d",batch)
                  display_sample.win=display.image( minibatch[1][{{1},{},{},{}}], display_sample) 
                  
                  local avg_val_error=0.0
                
                  local val_tp=0.0
                  local val_fp=0.0
                  local val_tn=0.0
                  local val_fn=0.0
                
                  local _minibatch={ minibatch[1]:narrow(1,1,1),
                                     minibatch[2]:narrow(1,1,1) }
                
                  model:evaluate()
                  local tiles=0
                
                  for i=1,validation do
                      get_sample( _minibatch, validation_dataset[i], mean_sd, qc_ref )
                      local outputs = model:forward(_minibatch[1])
                      avg_val_error=avg_val_error+criterion:forward(outputs, _minibatch[2])
                      local _,_outputs=outputs:max(2)
                    
                      if _minibatch[2][1]==2 then
                          if _outputs[{1,1}]==2 then 
                              val_tp=val_tp+1.0
                          else 
                              val_fn=val_fn+1.0 
                          end
                      else
                          if _outputs[{1,1}]==2 then 
                              val_fp=val_fp+1.0
                          else 
                              val_tn=val_tn+1.0 
                          end
                      end
                      -- val_acc=val_acc+accuracy_max(outputs, _minibatch[2])
                  end
                
                  model:training()
                  --print(val_tp,val_fp,val_fn,val_tn)
                  val_error=-math.log10(avg_val_error/validation)
                  val_acc=(val_tp+val_tn)/validation
                
                  if (val_tp+val_fn)>0 then val_tpr=(val_tp)/(val_tp+val_fn) else val_tpr=0.0 end
                  if (val_tn+val_fp)>0 then val_fpr=(val_fp)/(val_tn+val_fp) else val_fpr=0.0 end
              end
            
              table.insert(error_data,    {batch, avg_err, val_error})
              table.insert(accuracy_data, {batch, acc, val_acc, val_tpr, val_fpr })
            
              display_error.win    = display.plot(error_data,    display_error)
              display_accuracy.win = display.plot(accuracy_data, display_accuracy)
            
              iter_progress[#iter_progress+1]=
                { batch=batch,iter=p.iter+1,acc=acc,tpr=tpr,fpr=fpr,err=avg_err,
                  v_acc=val_acc,v_err=val_error,v_tpr=val_tpr,v_fpr=val_fpr}
            
              xlua.progress(batch,total_batches)
          end
      end
      save_progress(iter_progress,fold_prefix..'progress.txt')
      model:clearState()
      torch.save(final_model_name,model)
  end

  -- running testing
  do
      model:evaluate()
      collectgarbage()
    
      local _minibatch=allocate_minibatch(test_dataset, 1, qc_ref)
      local tiles=0

      local f = assert(io.open(fold_prefix..'test.csv', 'w'))
      f:write("id,fold,variant,cohort,subject,visit,truth,estimate,value\n")
      for i=1,#test_dataset do
          get_sample(_minibatch, test_dataset[i], mean_sd, qc_ref)
          local outputs = model:forward(_minibatch[1])
          local _,_outputs=outputs:max(2)
          f:write(string.format("%s,%d,%s,%s,%s,%s,%d,%d,%s\n", 
                  test_dataset[i][1], 
                  fold+1, 
                  test_dataset[i][4], test_dataset[i][5], test_dataset[i][6], test_dataset[i][7], test_dataset[i][2], _outputs[{1,1}]-1, outputs[{1,2}] ))
      end

      model:training()
  end
end
