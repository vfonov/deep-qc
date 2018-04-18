library(tidyverse)
library(pROC)
library(zoo)
library(DBI)
source("./multiplot.R")

con <- dbConnect(RSQLite::SQLite(), "data/qc_db.sqlite3")


pattern="results/r18_ref_%02d_%s"
method="results/r18_ref"
folds=8

cursor <- dbSendQuery(con, "SELECT q.variant,q.cohort,q.subject,q.visit,q.pass,lin,rx,ry,rz,tx,ty,tz,sx,sy,sz FROM qc_all as q left join xfm_dist as x on q.variant=x.variant and q.cohort=x.cohort and q.subject=x.subject and q.visit=x.visit ")
ref=dbFetch(cursor)

ref$id=paste(ref$variant,ref$cohort,ref$subject,ref$visit,sep='_')
ref$pass<-ref$pass=='TRUE'



l_test     <- vector(mode = "list", length = folds)
l_progress <- vector(mode = "list", length = folds)

for(i in seq(folds)) {
  l_test[[i]]=read_csv(sprintf(pattern,i,"test.csv"))
  l_test[[i]]$fold<-i
  l_test[[i]]$id<-paste(l_test[[i]]$variant,l_test[[i]]$cohort,l_test[[i]]$subject,l_test[[i]]$visit,sep='_')
  l_progress[[i]]=read_csv(sprintf(pattern,i,"progress.txt"))
  l_progress[[i]]$fold<-i
}

# rename variant and recreate id
test<-bind_rows(l_test) %>% mutate(
    variant=case_when(
    variant == "mi"            ~ "bestlinreg-mi",
    variant == "xcorr"         ~ "bestlinreg-xcorr",
    variant == "claude"        ~ "bestlinreg_claude",
    variant == "mritotal_std"  ~ "mritotal_std",
    variant == "mritotal_icbm" ~ "mritotal_icbm",
    variant == "elastix"       ~ "elastix"
    ),
    id=paste(variant,cohort,subject,visit,sep='_')
)




progress<-bind_rows(l_progress) %>% filter(v_accuracy>0)  # remove first entries when v_accuracy is set to 0

test<-merge(test,ref,by=c('id')) %>% mutate(
            pass=factor(pass, levels=c(T,F), labels=c('Pass','Fail')),
            result = as.factor(case_when(
                truth==1 & estimate==1 ~ "True\nPositive",
                truth==0 & estimate==0 ~ "True\nNegative",
                truth==0 & estimate==1 ~ "False\nPositive",
                truth==1 & estimate==0 ~ "False\nNegative"
                ))
            )

# suspicious results
ref_suspect <- test %>% filter(pass=='Pass', lin > 100)
write.csv(ref_suspect,file='r18_qc_suspect.csv',quote=F,row.names = F)


png(paste(method,"dx.png",sep="_"),width=1200,height=1200)

g_test<-test %>% gather(key=dx, value=measurement, rx,ry,rz, tx,ty,tz, sx,sy,sz) %>% 
    mutate(mes=factor(substr(dx,1,1),levels=c('r','t','s'),labels=c('Rotation','Translation','Scale')),
            ax=factor(substr(dx,2,2),levels=c('x','y','z'),labels=c('X','Y','Z')))

ggplot(g_test,aes(x=pass,y=measurement,colour=pass))+
    geom_violin()+
    facet_grid(mes~ax,scales='free')+
    theme(
      strip.text.x = element_text(size = 20,face = 'bold'),
      strip.text.y = element_text(size = 20,face = 'bold'),
      axis.text  = element_text(face = 'bold', vjust = 0.2, size = 20),
      axis.title = element_text(face = 'bold', vjust = 0.2, size = 22),
      legend.text= element_text(size=20),
      legend.position = 'none'
    )+xlab('')+ylab('')
    
    
fp_<-test %>% filter(truth==0 & estimate==1)
fn_<-test %>% filter(truth==1 & estimate==0)
tn_<-test %>% filter(truth==0 & estimate==0 & test$lin<5)

write.csv(fn_,file=paste(method,'fn.csv',sep="_"),quote=F,row.names = F)
write.csv(fp_,file=paste(method,'fp.csv',sep="_"),quote=F,row.names = F)
write.csv(tn_,file=paste(method,'tn_close.csv',sep="_"),quote=F,row.names = F)

png(paste(method,"dist_result.png",sep="_"),width=1000,height=700)

p1<-ggplot(test,aes(x=pass,y=lin))+
    geom_violin()+
    ylim(c(0,100))+
    theme(
      strip.text.x = element_text(size = 20,face = 'bold'),
      strip.text.y = element_text(size = 20,face = 'bold'),
      axis.text  = element_text(face = 'bold', vjust = 0.2, size = 20),
      axis.title = element_text(face = 'bold', vjust = 0.2, size = 22),
      legend.text= element_text(size=20),
      legend.position = 'none'
    )+ylab('Max distance (mm)')+xlab('Manual QC')

p2<-ggplot(test,aes(x=result,y=lin))+
    geom_violin()+
    ylim(c(0,100))+
    theme(
      strip.text.x = element_text(size = 20,face = 'bold'),
      strip.text.y = element_text(size = 20,face = 'bold'),
      axis.text    = element_text(face = 'bold', vjust = 0.2, size = 20),
      axis.title   = element_text(face = 'bold', vjust = 0.2, size = 22),
      legend.text  = element_text(size=20),
      legend.position = c(0.8, 0.2)
    )+ylab('')+xlab('ResNet-18')

multiplot(p1,p2,cols=2)
