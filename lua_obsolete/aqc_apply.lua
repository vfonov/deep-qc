#! /usr/bin/env th

require 'torch'
require 'xlua'
require 'paths'
require 'nn'
require 'image'
-- require 'minc2_simple'

torch.setdefaulttensortype('torch.FloatTensor')

function load_ref_data( img )
    local i,j
    local ret={}
    
    for i,j in ipairs(img) do
        ret[i]=image.load( j, 1, 'float' )
    end
    
    return ret
end

function parse(arg)
   local cmd = torch.CmdLine()
   local scriptdir=paths.dirname(paths.thisfile(arg))..'/..'
   
   cmd:text()
   cmd:text('Torch-based Auto QC application script ')
   cmd:text()
   cmd:text('Options:')
    ------------ General options --------------------
   cmd:option('-data',  scriptdir, 'Data prefix')
   cmd:option('-model', '',     'Alternative pre-trained model')
   cmd:option('-noref', false,  "Don't use reference")
   cmd:option('-gpu',   false,  'Use GPU/cudnn (model should be for GPU)')
   cmd:option('-volume', '',    'Input volume (minc)')
   cmd:option('-image', '',     'Input image base, will search for <base>_0.jpg, <base>_1.jpg, <base>_2.jpg ')
   cmd:option('-raw',  false,   'Print raw score [0:1]')
   cmd:option('-q',    false,   'Quiet mode, set status code to 0 - Pass, 1 - fail')
   cmd:text()
   local opt = cmd:parse(arg or {})
   
   return opt
end

local opt=parse(arg)

if not (opt.volume ~= "" or opt.image ~= "") then
    print(string.format("Need to specify input volume or image"))
    return 1
end

if opt.gpu then
    require 'cutorch'
    require 'cunn'
    require 'cudnn'
end

local model=nil
if opt.model ~= '' then
    model=torch.load(opt.model)
else
    model=torch.load(opt.data..'/results/r18_ref_01_training_cpu.t7')
end

local mean_sd=torch.load(opt.data..'/results/r18_ref_mean_sd.t7')

model:evaluate()

if not opt.noref then
qc_ref=load_ref_data({opt.data.."/data/mni_icbm152_t1_tal_nlin_sym_09c_0.jpg",
                      opt.data.."/data/mni_icbm152_t1_tal_nlin_sym_09c_1.jpg",
                      opt.data.."/data/mni_icbm152_t1_tal_nlin_sym_09c_2.jpg"})
end

if opt.image ~= "" then -- load JPGs from file
    input_image=load_ref_data({opt.image.."_0.jpg",opt.image.."_1.jpg",opt.image.."_2.jpg"} )
else
    input_minc=minc2_file.new(opt.volume)
    input_minc:setup_standard_order()

    -- load into standard volume
    local sample=input_minc:load_complete_volume(minc2_file.MINC2_FLOAT)
    
    -- normalize input
    local min=sample:min()
    local max=sample:max()
    sample=sample:add(-min):mul(1.0/(max-min))
    
    local sz=sample:size()
    
    input_image={sample[{sz[1]/2+1,{},{}}],
                 sample[{{},{},sz[3]/2+1}],
                 sample[{{},sz[2]/2+1,{}}]
                 }
    
    -- flip, resize and crop
    for i=1,3 do
        input_image[i]=image.scale(image.vflip(input_image[i]:contiguous()),256)
        sz=input_image[i]:size()
        
        --pad image 
        local dummy=torch.Tensor(256,256):zero()
        
        dummy[{ {(256-sz[1])/2+1, (256-sz[1])/2+sz[1]},{(256-sz[2])/2+1,(256-sz[2])/2+sz[2]}} ]=input_image[i]
        -- crop
        input_image[i]=image.crop(dummy,'c',224,224)
    end
end

if not opt.noref then
    data_in=torch.Tensor(     1, 6 , 224, 224 )
else
    data_in=torch.Tensor(     1, 3 , 224, 224 )
end

stride=1
if not opt.noref then stride=2 end

-- apply pre-scaling
for i=1,3 do
    input_image[i]:add( -mean_sd.mean  )
    input_image[i]:mul( 1.0/mean_sd.sd )
    
    
    data_in[{1,(i-1)*stride+1,{},{}}]=input_image[i]
    if not opt.noref then
        qc_ref[i]:add( -mean_sd.mean  )
        qc_ref[i]:mul( 1.0/mean_sd.sd )
        data_in[{1,(i-1)*stride+2,{},{}}]=qc_ref[i]
    end
end

-- get the result
local output = model:forward(data_in)
local _,out_max = output:max(2)
out_max=out_max:storage()[1]
out_p=output:exp():storage()[2]

if opt.raw then
    print(out_p)
elseif not opt.quiet then
    if out_max==2 then 
        print("PASS")
    else 
        print("FAIL")
    end
else
    -- return exit status
    os.exit(2-out_max)
end

-- 
