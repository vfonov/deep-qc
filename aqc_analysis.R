library(tidyverse)
library(stringr)
library(pROC)

#source("./multiplot.R")

inp=c(
     'results/r18_ref'
    ,'results/r18_noref'
    ,'results/nin_noref'
    ,'results/nin_ref'
    )
    
folds=8

lcv  <- vector(mode = "list", length = length(inp))
lcv2 <- vector(mode = "list", length = length(inp))


# alternative threshold
threshold=0.95

for(i in seq(length(inp))) {
    print(inp[i])
    dd  <- vector(mode = "list", length = folds)
    
    for(f in seq(folds)) {
        s=sprintf("%s_%02d_%s",inp[i],f,"test.csv")
        #print(s)
        
        dd[[f]]<-read_csv(s, 
        col_types=cols(
                id = col_character(),
                fold = col_integer(),
                variant = col_character(),
                cohort = col_character(),
                subject = col_character(),
                visit = col_character(),
                truth = col_integer(),
                estimate = col_integer(),
                value = col_double()
                )
            )
    }
    
    d<-bind_rows(dd)
    d$exp_value<-exp(d$value)
    method<-sub('results/','',inp[i])
    
    method_split<-str_split(method,'_')[[1]]
    
    r=roc(d$truth,d$exp_value,ci=F)
    
    # calculate sensetivity and specificity
    tp=sum(d$truth==1 & d$estimate==1)
    tn=sum(d$truth==0 & d$estimate==0)
    fp=sum(d$truth==0 & d$estimate==1)
    fn=sum(d$truth==1 & d$estimate==0)
    
    acc=(tp+tn)/length(d$truth)
    
    tpr=(tp)/(tp+fn)
    tnr=(tn)/(tn+fp)
    fpr=(fp)/(tn+fp)
    fnr=(fn)/(tn+fp)
    
    # adjusted estimate
    
    tp2=sum(d$truth==1 & d$exp_value >= threshold)
    tn2=sum(d$truth==0 & d$exp_value <  threshold)
    fp2=sum(d$truth==0 & d$exp_value >= threshold)
    fn2=sum(d$truth==1 & d$exp_value <  threshold)
    
    acc2=(tp2+tn2)/length(d$truth)
    
    tpr2=(tp2)/(tp2+fn2)
    tnr2=(tn2)/(tn2+fp2)
    fpr2=(fp2)/(tn2+fp2)
    fnr2=(fn2)/(tn2+fp2)
    
    
    
    lcv[[i]]=data.frame(
        sen=r$sen,
        spe=r$spe,
        method=method,
        net=method_split[1],
        ver=method_split[2]
    )
    
    lcv2[[i]]=data.frame(
        acc=acc,
        sen=tpr,
        spe=tnr,
        
        tpr=tpr,
        tnr=tnr,
        fpr=fpr,
        fnr=fnr,

        acc2=acc2,
       
        tpr2=tpr2,
        tnr2=tnr2,
        fpr2=fpr2,
        fnr2=fnr2,
        
        auc=r$auc+0.0,
        
        method=method,
        net=method_split[1],
        ver=method_split[2],
        auc_=paste(method,format(r$auc,digits=4),sep=' ')
    )
}

cv <-bind_rows(lcv) %>% mutate (
        ver=factor(ver,levels=c('ref','noref'),labels=c('With Reference','No Reference')),
        net=factor(net,levels=c('r18','nin'),  labels=c('Resnet 18','Network-in-Network'))
)

cv2<-bind_rows(lcv2) %>% mutate (
        Auc=sprintf("AUC=%0.4f", auc),
        ver=factor(ver,levels=c('ref','noref'),labels=c('With Reference','No Reference')),
        net=factor(net,levels=c('r18','nin'),  labels=c('Resnet 18','Network-in-Network'))
    )


png("results/auc.png",width=1200,height=800)

ggplot(cv,aes(x=1-spe, y=sen, col=method))+
    theme_bw()+
    theme(
      strip.text.x = element_text(size = 20,face = 'bold'),
      strip.text.y = element_text(size = 20,face = 'bold'),
      axis.text    = element_text(face = 'bold', vjust = 0.2, size = 20),
      axis.text.x  = element_text(  angle = 90,  hjust = 1,   size = 20, vjust = 0.2),
      axis.title   = element_text(face = 'bold', vjust = 0.2, size = 22),
      plot.margin  = unit(c(0.2,0.2,0.2,0.2), "cm"),
      legend.text  = element_text(face = 'bold', vjust = 0.2, size = 22),
      legend.position="none"
    )+
    geom_line(size=2)+
    geom_label(data=cv2,aes(x=0.5, y=0.5, label=Auc, size=4,fontface = "bold"))+
    facet_grid(net~ver)+
    scale_colour_manual(
      breaks = cv2$method,
      labels = cv2$auc_,
      values=scales::hue_pal()(length(inp))
    )

ev=stack(cv2, select=c('acc','tpr','fpr','auc'))
names(ev)=c('Value','Measure')

ev$method=rep(cv2$method,4)
ev$net=rep(cv2$net,4)
ev$ver=rep(cv2$ver,4)

ev$Val=format(ev$Value,digits=4)
print(ev)


png("results/acc_tpr_fpr_auc.png",width=1200,height=800)

ggplot(ev,aes(x=Measure,y=Value,fill=Measure,label=Val))+
    theme_bw()+
    theme(
      strip.text.x = element_text(size = 20,face = 'bold'),
      strip.text.y = element_text(size = 20,face = 'bold'),
      axis.text    = element_text(face = 'bold', vjust = 0.2, size = 20),
      axis.text.x  = element_text(angle = 90,    hjust = 1,   size = 20,vjust = 0.2),
      axis.title   = element_text(face = 'bold', vjust = 0.2, size = 22),
      plot.margin  = unit(c(0.2,0.2,0.2,0.2), "cm"),
      legend.text  = element_text(face = 'bold', vjust = 0.2, size = 22),
      legend.position="none"
    )+
    geom_bar(stat='identity')+
    facet_grid(net~ver)+
    geom_label(vjust=1,size=6)


ev=stack(cv2, select=c('acc2','tpr2','fpr2'))
names(ev)=c('Value','Measure')

ev$method<-rep(cv2$method,3)
ev$Measure<-factor(ev$Measure,levels=c('acc2','tpr2','fpr2'),labels=c('acc','tpr','fpr'))
ev$net<-rep(cv2$net,3)
ev$ver<-rep(cv2$ver,3)

ev$Val=format(ev$Value,digits=4)
print(ev)
    
png("results/acc_tpr_fpr_auc2.png",width=1000,height=800)

ggplot(ev,aes(x=Measure,y=Value,fill=Measure,label=Val))+
    theme_bw()+
    theme(
      strip.text.x = element_text(size = 14),
      strip.text.y = element_text(size = 14),
      axis.text    = element_text(face = 'bold', vjust = 0.2, size = 20),
      axis.text.x  = element_text(  angle = 90,  hjust = 1,   size = 20,vjust = 0.2),
      axis.title   = element_text(face = 'bold', vjust = 0.2, size = 22),
      plot.margin  = unit(c(0.2,0.2,0.2,0.2), "cm"),
      legend.text  = element_text(face = 'bold', vjust = 0.2, size = 22),
      legend.position="none"
    )+
    geom_bar(stat='identity')+
    facet_grid(net~ver)+
    geom_label(vjust=1,size=5)
