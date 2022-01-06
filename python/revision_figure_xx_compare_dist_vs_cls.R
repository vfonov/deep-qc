library(tidyverse)
library(jsonlite)
library(pROC)


v='r152'
r='_ref'

cv<-data.frame()

lr_='0.0001'
# load QCcls :
pfx=paste0('cls/lr_',lr_,'_pre/')

for(f in seq(0,7)) {
    fn=paste0(pfx,'model_',v,r,'/log_',f,'_8.json')
    if(file.exists(fn)) {
        rrr=fromJSON(fn)

        # as.data.frame(rrr$testing_best_tnr$summary)%>%mutate(select_kind='tnr')) %>%
        # mutate(fold=f,ref=(r=="_ref"),model=v,lr=lr_, pre=T,flavor='CLS')
        # cv<-bind_rows(cv,fold)

        best<-as.data.frame(rrr$testing_best_tnr$details)
        # calculate AUC, TODO: calculate ACC?
        r_best<-roc(best$labels, best$scores, stratified=F, auc=T)

        fold <- as.data.frame(rrr$testing_best_tnr$summary) %>% 
                mutate(select_kind='tnr', auc=as.numeric(r_best$auc)) %>%
                mutate(fold=f,ref=(r=="_ref"),model=v,lr=lr_, pre=T,flavor='QCResNet152')

        cv<-bind_rows(cv,fold)

    }
    else {
        print(paste("Missing",fn))
    }
}

# load QCdist
pfx=paste0('dist/lr_',lr_,'_pre/')
for(f in seq(0,7)) {
    fn=paste0(pfx,'model_',v,r,'/log_',f,'_8.json')
    if(file.exists(fn)) {
        rrr=fromJSON(fn)

        best<-as.data.frame(rrr$testing_best_loss$details)
        # calculate AUC, TODO: calculate ACC?
        r_best<-roc(best$labels, best$preds, stratified=F, auc=T)

        fold <- as.data.frame(rrr$testing_best_loss$summary) %>% 
                mutate(select_kind='loss', auc=as.numeric(r_best$auc)) %>%
                mutate(fold=f,ref=(r=="_ref"),model=v,lr=lr_, pre=T,flavor='DistResNet152')

        cv<-bind_rows(cv,fold)
        }
        else {
            print(paste("Missing",fn))
        }
}

cv<-cv%>%mutate(flavor=as.factor(flavor))
print(summary(cv$flavor))

tccv<-cv %>% group_by(flavor) %>% 
  summarize( auc=median(auc),auc_lab=signif(auc,4)) %>% ungroup()


p<-ggplot(cv,aes(y=auc, x=flavor))+
    theme_bw(base_size = 28)+
    geom_violin(trim = FALSE,alpha=0.8)+
    stat_summary(
        fun.data = "median_hilow", 
        geom = "pointrange", color = "black"
        )+
    ylab('')+xlab('')+
    geom_text(data=tccv, aes(label=auc_lab, x=flavor, y=auc), 
              show.legend = F, size=6, nudge_y=0.005)+
    ggtitle("Area under the curve")

png("Figure_XX_auc_compare.png", 
    width=10, height=10, res=200, units = "in", 
    pointsize = 12, type='cairo', antialias = "default")

print(p)


# do statistical comparison

print(wilcox.test(auc~flavor,data=cv))