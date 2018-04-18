library(tidyverse)
library(pROC)
library(zoo)

source("./multiplot.R")

pattern="results/r18_noref_long_%02d_%s"
method="results/r18_noref_long"
folds=1

l_test     <- vector(mode = "list", length = folds)
l_progress <- vector(mode = "list", length = folds)

for(i in seq(folds)) {
  l_test[[i]]=read_csv(sprintf(pattern,i,"test.csv"))
  l_test[[i]]$fold<-i
  l_test[[i]]$id<-paste(l_test[[i]]$variant,l_test[[i]]$cohort,l_test[[i]]$subject,l_test[[i]]$visit,sep='_')
  l_progress[[i]]=read_csv(sprintf(pattern,i,"progress.txt"))
  l_progress[[i]]$fold<-i
}

test<-bind_rows(l_test)

progress<-l_progress[[1]]
progress<-progress[progress$v_accuracy>0,] # remove first entries when v_accuracy is set to 0

r=roc(test$truth,exp(test$value),ci=T)

# calculate sensetivity and specificity
tp=sum(test$truth==1 &test$estimate==1)
tn=sum(test$truth==0 &test$estimate==0)
fp=sum(test$truth==0 &test$estimate==1)
fn=sum(test$truth==1 &test$estimate==0)

acc=(tp+tn)/length(test$truth)

tpr=(tp)/(tp+fn)
tnr=(tn)/(tn+fp)

fpr=(fp)/(tn+fp)
fnr=(fn)/(tn+fp)

cv=data.frame(
    sen=r$sen,
    spe=r$spe,
    method=method
)

cv2=data.frame(
    acc=acc,
    
    tpr=tpr,
    fpr=fpr,
    
    method=method,
    auc=paste(method,format(r$auc,digits=4),sep=' ')
)

ev=stack(cv2,select=c('acc','tpr','fpr' ))
names(ev)=c('Value','Measure')
ev$Method=rep(cv2$method,3)
ev$Val=format(ev$Value,digits=4)


png(paste(method,"progress.png",sep="_"),width=1600,height=800)

bw=200
progress$r_accuracy<-rollmean(progress$accuracy,bw,fill="extend")
progress$r_v_accuracy<-rollmean(progress$v_accuracy,bw,fill="extend")
progress$r_v_fpr<-rollmean(progress$v_fpr,bw,fill="extend")
progress$r_v_error<-rollmean(progress$v_error,bw,fill="extend")
progress$r_error<-rollmean(progress$error,bw,fill="extend")


p1<-ggplot(progress,aes(x=batch,y=r_accuracy,col='Training Accuracy'))+
    theme_bw()+
    theme(
      axis.text =  element_text(face = 'bold', vjust = 0.2, size = 20),
      axis.title = element_text(face = 'bold', vjust = 0.2, size = 22),
      legend.text= element_text(face = 'bold', vjust = 0.2, size = 22),
      legend.position=c(0.6,0.5),
      plot.margin = unit(c(0.2,0.2,0.2,0.2), "cm")
    )+
  ylab('')+
  xlab('minibatch')+
  geom_line(alpha=1.0)+
  geom_line(aes(x=batch,y=r_v_accuracy,col='Validation Accuracy'),alpha=1.0)+
  geom_line(aes(x=batch,y=r_v_fpr,col='Validation FPR'),alpha=1.0)+
  scale_colour_manual("",values = c("Training Accuracy"='green',"Validation Accuracy"='red',"Validation FPR"='blue'))

p2<-ggplot(progress,aes(x=batch,y=error,col='Training'))+
    theme_bw()+
    theme(
      axis.text =  element_text(face = 'bold', vjust = 0.2, size = 20),
      axis.title = element_text(face = 'bold', vjust = 0.2, size = 22),
      plot.margin = unit(c(0.2,0.2,0.2,0.2), "cm"),
      legend.text= element_text(face = 'bold', vjust = 0.2, size = 22),
      legend.position=c(0.8,0.3)
    )+
  ylab('-10*log(loss)')+
  xlab('minibatch')+
  geom_line(aes(x=batch,y=r_error),alpha=1.0)+
  geom_line(aes(x=batch,y=r_v_error,col='Validation'),alpha=1.0)+
  scale_colour_manual("",values = c("Training"='green',"Validation"='red'))

multiplot(p1,p2,cols=2)
